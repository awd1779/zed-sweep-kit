#!/usr/bin/env python3
"""
zed_capture_session.py -- record a lab-mapping session with a ZED 2 / ZED 2i.

Produces, inside --outdir (default ./captures/):

  zed_session_<YYYYMMDD_HHMMSS>/
    camera_info.json          rectified intrinsics of the OPENED camera + baseline
                              + depth_mode / depth_units / depth_aligned_to
    images/left/000000.jpg    rectified left frames,  JPEG quality 95, 6-digit indices
    images/right/000000.jpg   rectified right frames, same indices (stereo pairs)
    depth/000000.png          SDK depth (sl.MEASURE.DEPTH, aligned to LEFT camera),
                              uint16 PNG in MILLIMETRES, same indices as images/left;
                              invalid / occluded / NaN / inf pixels -> 0
    poses.txt                 "# timestamp tx ty tz qx qy qz qw" then one TUM line per
                              frame; camera-to-world, metric, timestamp in seconds
    imu.csv                   timestamp_ns,ax,ay,az,gx,gy,gz
                              (accel m/s^2 incl. gravity, gyro deg/s -- SDK native units)
    timestamps.csv            frame_index,timestamp_ns,tracking_state
    recording.svo2            master SVO2 recording (everything above is re-derivable);
                              H264/H265. Skipped -- not LOSSLESS -- if the GPU encoder
                              fails (--allow-lossless-svo overrides). Optional:
                              skip with --no-svo (dumps only, not re-derivable)
    area.area                 spatial-memory map for relocalisation (--no-save-area)
    keyframes.txt             GEN_3 keyframe poses, for comparison with poses.txt
    capture.log               capture warnings/summary (not consumed by the pipeline)

Conventions:
  * Images are the RECTIFIED pair (sl.VIEW.LEFT / RIGHT); the intrinsics in
    camera_info.json are the rectified model, so distortion is ~0.
  * Poses use sl.COORDINATE_SYSTEM.IMAGE (x right, y down, z forward -- OpenCV
    convention), sl.UNIT.METER, world frame anchored where tracking is enabled
    (i.e. at the start of the recording, after a short auto-exposure warm-up).
  * Depth runs at DEPTH_MODE.NEURAL, falling back to ULTRA only if NEURAL cannot
    run on this machine. Every grabbed frame gets its L/R/depth triple -- if the
    machine cannot sustain the target fps the achieved rate is reported honestly
    (SDK frame gaps are logged); stereo frames are never dropped to keep depth.
  * Frame indices are contiguous over the frames that were actually grabbed;
    a failed grab is logged and does NOT consume an index.

A live preview window (left camera, half size, REC/fps/tracking overlay) is
shown while recording; press q or Esc in it to stop -- same clean shutdown as
Ctrl-C. Closing the window with X only hides the preview; recording continues.
Pass --no-preview for headless use.

Stop with Ctrl-C (flushes and closes everything cleanly) or pass --duration.

Examples:
  python scripts/zed_capture_session.py --duration 20            # smoke test
  python scripts/zed_capture_session.py                          # until Ctrl-C
  python scripts/zed_capture_session.py --outdir D:\\zed_captures --fps 15 --res HD720
"""

import argparse
import json
import math
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

JPEG_QUALITY = 95                    # spec requires >= 95
PNG_COMPRESSION = 1                  # fast; depth PNGs are throughput-critical
N_WRITE_WORKERS = 4                  # 2 JPEGs + 1 depth PNG (incl. conversion) per frame
WARMUP_FRAMES = 15                   # ~1 s @15fps for auto exposure/gain to settle
TRACKER_START_TIMEOUT = 20           # s to wait for the tracker to go live
WRITE_QUEUE_MAX = 60                 # ~20 frames buffered; submit() blocks beyond
MAX_CONSECUTIVE_GRAB_FAILURES = 150  # ~10 s of failures => abort (cable pulled etc.)
PREVIEW_WINDOW = "ZED capture  [q / Esc = stop recording]"

RES_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}
SVO_COMPRESSION_MAP = {
    "h264": sl.SVO_COMPRESSION_MODE.H264,
    "h265": sl.SVO_COMPRESSION_MODE.H265,
    "h264_lossless": sl.SVO_COMPRESSION_MODE.H264_LOSSLESS,
    "h265_lossless": sl.SVO_COMPRESSION_MODE.H265_LOSSLESS,
    "lossless": sl.SVO_COMPRESSION_MODE.LOSSLESS,  # CPU-side; no NVENC needed
}
STOP = threading.Event()
_LOGFILE = None
_WRITER = None          # set once the image writer exists, for the SIGINT handler
_SIGINT_COUNT = 0
FORCE_QUIT_PRESSES = 5  # Ctrl-C presses needed to abandon queued writes


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGFILE is not None:
        _LOGFILE.write(line + "\n")
        _LOGFILE.flush()


def enum_name(v) -> str:
    """'OK', 'SEARCHING_FLOOR_PLANE', ... regardless of pyzed __str__ quirks."""
    n = getattr(v, "name", None)
    if isinstance(n, str) and n:
        return n
    return str(v).replace(" ", "_")


def depth_mode_candidates(arg):
    """Required: NEURAL, falling back to ULTRA only if NEURAL can't run here.
    An explicit --depth-mode disables the fallback."""
    if arg is not None:
        mode = getattr(sl.DEPTH_MODE, arg.upper(), None)
        if mode is None:
            sys.exit(f"FATAL: depth mode {arg!r} not available in this SDK build")
        return [mode]
    return [getattr(sl.DEPTH_MODE, n) for n in ("NEURAL", "ULTRA")
            if hasattr(sl.DEPTH_MODE, n)]


def depth_to_u16_mm(depth_m: np.ndarray) -> np.ndarray:
    """float32 metres -> uint16 millimetres; NaN/inf/<=0/overflow -> 0."""
    mm = depth_m * 1000.0
    valid = np.isfinite(mm) & (mm > 0) & (mm <= 65535.0)
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    out[valid] = np.rint(mm[valid]).astype(np.uint16)
    return out


def lower_thread_priority() -> None:
    """Run bulk image encoding below normal priority so the SDK's own threads
    win CPU contention. GEN_3 tracks on the CPU, and it froze once while the
    encoders saturated it. Windows only; a no-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentThread.restype = wintypes.HANDLE
        k32.SetThreadPriority.argtypes = (wintypes.HANDLE, ctypes.c_int)
        k32.SetThreadPriority.restype = wintypes.BOOL
        k32.SetThreadPriority(k32.GetCurrentThread(), -1)  # BELOW_NORMAL
    except Exception:
        pass


def export_area_map(zed: sl.Camera, path: Path) -> None:
    """Save GEN_3's spatial memory. The export is asynchronous: grab() keeps it
    progressing, and closing the camera before it finishes truncates the file."""
    log("Saving spatial-memory map to area.area -- this can take up to a "
        "minute. Do NOT press Ctrl-C.")
    if zed.save_area_map(str(path)) != sl.ERROR_CODE.SUCCESS:
        log("WARNING: save_area_map was refused")
        return
    rt, t0 = sl.RuntimeParameters(), time.time()
    while (zed.get_area_export_state() == sl.AREA_EXPORTING_STATE.RUNNING
           and time.time() - t0 < 180):
        zed.grab(rt)
        time.sleep(0.05)
    size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0
    log(f"area map    : {enum_name(zed.get_area_export_state())} "
        f"({size_mb:.1f} MB)")


class Preview:
    """Live view of the left camera while recording.

    Prefers cv2's HighGUI, but falls back to Tk + Pillow when the installed
    OpenCV has no GUI support -- opencv-python-headless shadows the full build
    in some environments, and then imshow raises "function is not implemented".
    (cv2's drawing/resize calls still work headless; only windowing is missing.)
    show() returns 'stop' when the user asks to end the capture, 'closed' when
    they dismissed the window, else None."""

    def __init__(self, width: int, height: int):
        self.backend = None
        self._tk = None
        self._stop = False
        self._closed = False
        # Tk+PIL costs far more per frame than cv2's imshow, so its preview runs
        # smaller and slower. Both are plenty for framing and status on a rig.
        self.scale, self.target_hz = 2, 15.0
        try:
            # NORMAL (not AUTOSIZE) so it can be resized on the rig; TOPMOST so
            # it does not hide behind the terminal on Windows.
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW, width // self.scale,
                             height // self.scale)
            cv2.moveWindow(PREVIEW_WINDOW, 40, 40)
            try:
                cv2.setWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_TOPMOST, 1)
            except (cv2.error, AttributeError):
                pass  # older OpenCV: window works, just not always-on-top
            self.backend = "cv2"
            return
        except cv2.error:
            pass

        import tkinter as tk
        from PIL import Image, ImageTk
        self.scale, self.target_hz = 4, 5.0
        self._Image, self._ImageTk = Image, ImageTk
        root = tk.Tk()
        root.title(PREVIEW_WINDOW)
        root.geometry(f"{width // self.scale}x{height // self.scale}+40+40")
        root.attributes("-topmost", True)
        self._label = tk.Label(root)
        self._label.pack()
        root.bind("<Key>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tk, self._photo = root, None
        self.backend = "tk"

    def _on_key(self, event) -> None:
        if event.keysym in ("Escape", "q", "Q"):
            self._stop = True

    def _on_close(self) -> None:
        self._closed = True

    def show(self, bgr):
        if self.backend == "cv2":
            cv2.imshow(PREVIEW_WINDOW, bgr)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                return "stop"
            if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return "closed"
            return None
        if self._closed or self._tk is None:
            return "closed"
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # PIL wants RGB, positive strides
        self._photo = self._ImageTk.PhotoImage(self._Image.fromarray(rgb))
        self._label.configure(image=self._photo)
        self._tk.update_idletasks()
        self._tk.update()
        if self._stop:
            return "stop"
        return "closed" if self._closed else None

    def close(self) -> None:
        if self.backend == "cv2":
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        elif self._tk is not None:
            tkroot, self._tk = self._tk, None
            try:
                self._photo = None      # drop image refs before tearing down Tcl
                tkroot.quit()
                tkroot.update()
                tkroot.destroy()
            except Exception:
                pass
        self._tk = None


class AsyncJpegWriter:
    """Encodes/writes JPEGs off the grab thread so the grab loop keeps up with
    the camera. submit() blocks when the queue is full (bounded memory); the
    resulting camera-frame gap is detected by the main loop's timestamp check
    and logged -- nothing is dropped silently."""

    def __init__(self, n_workers: int = N_WRITE_WORKERS):
        self._q = queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self._lock = threading.Lock()
        self.written = 0
        self.failed = []
        self._workers = [threading.Thread(target=self._worker, daemon=True)
                         for _ in range(n_workers)]
        for w in self._workers:
            w.start()

    def submit(self, path: Path, bgra) -> None:
        self._q.put((path, bgra))

    def _worker(self) -> None:
        lower_thread_priority()
        while True:
            item = self._q.get()
            if item is None:
                break
            path, img = item
            try:
                if path.suffix == ".png":      # depth: float32 m -> uint16 mm
                    if img.dtype != np.uint16:
                        img = depth_to_u16_mm(img)
                    written = cv2.imwrite(str(path), img,
                                          [int(cv2.IMWRITE_PNG_COMPRESSION),
                                           PNG_COMPRESSION])
                else:                          # BGRA view -> BGR JPEG
                    written = cv2.imwrite(str(path),
                                          cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
                                          [int(cv2.IMWRITE_JPEG_QUALITY),
                                           JPEG_QUALITY])
                if not written:
                    raise IOError("cv2.imwrite returned False")
                with self._lock:
                    self.written += 1
            except Exception as exc:  # keep worker alive; record the failure
                with self._lock:
                    self.failed.append(str(path))
                log(f"ERROR: failed to write {path}: {exc}")

    def qsize(self) -> int:
        return self._q.qsize()

    def close(self) -> None:
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            w.join()


def pose_key(tr, q) -> tuple:
    """Pose rounded to 1e-7. A live tracker never repeats it exactly (even a
    camera sitting still jitters), so an identical key on consecutive frames
    means the tracker is not updating."""
    return tuple(round(float(v), 7) for v in (*tr[:3], *q[:4]))


def write_imu_batch(zed: sl.Camera, f, last_ts: int):
    """Append every IMU sample queued since the previous grab (~27 per frame at
    400 Hz). Reading them on the grab thread replaces a polling thread that
    missed samples (~325 Hz) and cost ~25% frame rate. Returns the newest
    timestamp written and the number of rows."""
    batch = []
    if zed.get_sensors_data_batch(batch) != sl.ERROR_CODE.SUCCESS:
        return last_ts, 0
    rows = 0
    for sd in batch:
        imu = sd.get_imu_data()
        ts = imu.timestamp.get_nanoseconds()
        if ts <= last_ts:                   # consecutive batches can overlap
            continue
        a = imu.get_linear_acceleration()   # m/s^2, gravity included
        g = imu.get_angular_velocity()      # deg/s
        f.write(f"{ts},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f},"
                f"{g[0]:.6f},{g[1]:.6f},{g[2]:.6f}\n")
        last_ts = ts
        rows += 1
    return last_ts, rows


def preview_status(preview, zed: sl.Camera, mat: sl.Mat, text: str, color):
    """Show the live left view with one status line; returns Preview.show()."""
    zed.retrieve_image(mat, sl.VIEW.LEFT)
    view = mat.get_data(deep_copy=False)
    small = cv2.cvtColor(cv2.resize(view, (view.shape[1] // preview.scale,
                                           view.shape[0] // preview.scale)),
                         cv2.COLOR_BGRA2BGR)
    cv2.putText(small, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA)
    return preview.show(small)


def build_camera_info(zed: sl.Camera, requested_fps: int, depth_mode) -> dict:
    info = zed.get_camera_information()
    conf = info.camera_configuration
    calib = conf.calibration_parameters  # RECTIFIED model (raw variant not used)

    def cam_dict(cam) -> dict:
        d = [float(x) for x in list(cam.disto)[:5]]  # [k1,k2,p1,p2,k3], ~0 rectified
        return {"fx": float(cam.fx), "fy": float(cam.fy),
                "cx": float(cam.cx), "cy": float(cam.cy),
                "k1": d[0], "k2": d[1], "p1": d[2], "p2": d[3], "k3": d[4]}

    try:
        baseline = float(calib.get_camera_baseline())  # coordinate_units=METER -> m
    except Exception:
        baseline = float(abs(calib.stereo_transform.get_translation().get()[0]))

    return {
        "model": str(info.camera_model),               # "ZED 2" / "ZED 2i"
        "serial_number": int(info.serial_number),
        "firmware": int(conf.firmware_version),
        "resolution": {"width": int(conf.resolution.width),
                       "height": int(conf.resolution.height)},
        "fps": float(conf.fps if conf.fps > 0 else requested_fps),
        "baseline_m": baseline,
        "left_camera": cam_dict(calib.left_cam),
        "right_camera": cam_dict(calib.right_cam),
        "depth_mode": enum_name(depth_mode),
        "depth_units": "mm",
        "depth_aligned_to": "left",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Record a ZED 2 session (SVO2 master + per-frame dumps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many seconds (default: run until Ctrl-C)")
    ap.add_argument("--outdir", default="./captures",
                    help="parent directory for zed_session_* folders")
    ap.add_argument("--fps", type=int, default=15,
                    help="camera fps (HD720 supports 15/30/60)")
    ap.add_argument("--res", choices=sorted(RES_MAP), default="HD720",
                    help="camera resolution")
    ap.add_argument("--svo-compression", choices=sorted(SVO_COMPRESSION_MAP),
                    default="h264", help="SVO2 compression codec")
    ap.add_argument("--exposure", type=int, default=None, metavar="PCT",
                    help="lock exposure at this %% of the frame period and let "
                         "gain compensate (ZED 2i over USB has no microsecond "
                         "control). At 15 fps: 15%% = 10 ms, 22%% = 15 ms. "
                         "Lower = less motion blur, more noise. The single best "
                         "lever against blur; also stops brightness pumping.")
    ap.add_argument("--gain", type=int, default=None, metavar="PCT",
                    help="gain to pair with --exposure (default: leave the "
                         "camera's current value)")
    ap.add_argument("--lock-wb", type=int, default=None, metavar="KELVIN",
                    help="lock white balance at this colour temperature (e.g. "
                         "4000 for lab fluorescents); stops colour pumping "
                         "between frames, which hurts feature matching")
    ap.add_argument("--no-svo", action="store_true",
                    help="skip the recording.svo2 master copy (per-frame dumps "
                         "only; saves ~2 MB/frame while NVENC is broken)")
    ap.add_argument("--allow-lossless-svo", action="store_true",
                    help="permit the LOSSLESS SVO fallback when the GPU encoder "
                         "is unavailable. OFF by default: LOSSLESS runs ~4 MB per "
                         "frame at HD1080, and SVO2 files past ~4 GB are known to "
                         "corrupt on Windows 11 -- a long sweep produces a file "
                         "that cannot be reopened. Fine for captures under ~1 min.")
    ap.add_argument("--no-preview", action="store_true",
                    help="disable the live preview window (headless/remote use)")
    ap.add_argument("--tracking-mode", default="gen3",
                    choices=["gen3", "gen1", "gen2"],
                    help="ZED positional tracking engine. GEN_3 = feature-based "
                         "visual-inertial SLAM with loop closure (Stereolabs' most "
                         "accurate). GEN_1 = legacy depth-driven VIO. On SDK "
                         "5.2.0 GEN_3 froze once under heavy CPU load -- SDK 5.5 "
                         "fixes GEN_3 stalls; the freeze guard below protects "
                         "the walk either way. GEN_2 is deprecated.")
    ap.add_argument("--track-compute", default="gpu",
                    choices=["gpu", "auto", "cpu"],
                    help="where GEN_3 runs its tracking work (ZED SDK >= 5.5). "
                         "gpu keeps it off the CPU, which the image encoders "
                         "load heavily -- GEN_3 froze once when the CPU was "
                         "saturated. Try auto if the frame rate drops.")
    ap.add_argument("--no-freeze-stop", action="store_true",
                    help="keep recording if the tracker freezes (default: stop "
                         "automatically after 3 s frozen, so you don't walk on "
                         "with dead poses)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the automatic zed_verify_session.py run at the end")
    ap.add_argument("--no-save-area", action="store_true",
                    help="do not save the spatial-memory map (area.area) at the "
                         "end. The .area lets later sessions -- or Spot at "
                         "runtime -- relocalise into this session's frame.")
    ap.add_argument("--depth-mode", default=None,
                    choices=["neural", "ultra", "neural_plus", "neural_light",
                             "quality", "performance"],
                    help="force an SDK depth mode (default: NEURAL, falling back "
                         "to ULTRA only if NEURAL cannot run on this machine)")
    return ap.parse_args()


def main() -> int:
    global _LOGFILE
    args = parse_args()
    args.tracking_mode = {"gen1": "GEN_1", "gen2": "GEN_2",
                          "gen3": "GEN_3"}[args.tracking_mode]

    def _sigint(_sig, _frm):
        # Extra Ctrl-C presses must NOT kill a drain in progress: the queue can
        # hold ~20 frames, and quitting there silently truncates the session
        # (that is how a session once lost its last depth frame).
        global _SIGINT_COUNT
        _SIGINT_COUNT += 1
        if not STOP.is_set():
            STOP.set()
            log("Ctrl-C -- stopping capture, flushing writers ...")
            return
        pending = _WRITER.qsize() if _WRITER is not None else 0
        left = FORCE_QUIT_PRESSES - _SIGINT_COUNT
        if left <= 0:
            log(f"Force-quit: ABANDONING {pending} queued file(s) -- this "
                f"session will be INCOMPLETE.")
            sys.exit(130)
        log(f"Still flushing ({pending} file(s) queued) -- hold on. Press "
            f"Ctrl-C {left} more time(s) to abandon them and truncate the "
            f"session.")

    signal.signal(signal.SIGINT, _sigint)

    candidates = depth_mode_candidates(args.depth_mode)
    zed = sl.Camera()
    err = None
    depth_mode = None
    for i, mode in enumerate(candidates):
        print(f"Opening camera ({args.res} @ {args.fps} fps, "
              f"depth={enum_name(mode)}) ...", flush=True)
        if enum_name(mode) == "NEURAL":
            print("(first NEURAL run can take minutes while the SDK downloads/"
                  "optimizes its AI model -- later runs are instant)", flush=True)
        init = sl.InitParameters(
            camera_resolution=RES_MAP[args.res],
            camera_fps=args.fps,
            depth_mode=mode,
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.IMAGE,  # x right, y down, z fwd
        )
        err = zed.open(init)
        if err == sl.ERROR_CODE.SUCCESS:
            depth_mode = mode
            break
        if i + 1 < len(candidates):
            print(f"open with depth={enum_name(mode)} failed ({err}) -- "
                  f"falling back to {enum_name(candidates[i + 1])} ...", flush=True)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"FATAL: camera open failed: {err}", file=sys.stderr)
        print("Hints: use a USB 3.0 port (direct, no hub); reseat the cable; close\n"
              "ZED Explorer or anything else using the camera; run\n"
              "'C:\\Program Files (x86)\\ZED SDK\\tools\\ZED Diagnostic.exe'.",
              file=sys.stderr)
        return 2

    session_name = f"zed_session_{datetime.now():%Y%m%d_%H%M%S}"
    session_dir = Path(args.outdir).expanduser() / session_name
    left_dir = session_dir / "images" / "left"
    right_dir = session_dir / "images" / "right"
    depth_dir = session_dir / "depth"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    _LOGFILE = open(session_dir / "capture.log", "w", encoding="utf-8", newline="\n")

    poses_f = ts_f = None
    writer = imu_f = preview = None
    imu_last, imu_rows = -1, 0
    recording_enabled = tracking_enabled = False
    fatal = None

    idx = 0
    first_ts = last_ts = None
    ok_frames = grab_failures = gap_events = dup_frames = 0
    loop_closures = loop_flash = insufficient_feat = 0
    # A hand-held camera never holds a bit-identical pose, so a run of
    # identical poses means the tracker has stopped updating -- whatever
    # tracking_state claims. (GEN_3 on SDK 5.2.0 did exactly that.)
    prev_pose_key, frozen_run, frozen_frames, freeze_events = None, 0, 0, 0
    last_mem, fusion_name, odo, mem = None, "?", "?", "?"
    state_counts = {}

    try:
        cam_info = build_camera_info(zed, args.fps, depth_mode)
        with open(session_dir / "camera_info.json", "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump(cam_info, f, indent=2)
            f.write("\n")

        log(f"Session dir : {session_dir.resolve()}")
        log(f"Camera      : {cam_info['model']}  SN {cam_info['serial_number']}  "
            f"FW {cam_info['firmware']}")
        log(f"Rectified   : {cam_info['resolution']['width']}x"
            f"{cam_info['resolution']['height']} @ {cam_info['fps']:g} fps, "
            f"baseline {cam_info['baseline_m']:.4f} m")
        log(f"Depth       : {cam_info['depth_mode']} -> uint16 mm PNGs aligned "
            f"to left camera")
        if int(cam_info["fps"]) != args.fps:
            log(f"WARNING: SDK coerced fps {args.fps} -> {cam_info['fps']:g} "
                f"(not all fps are valid at {args.res})")

        # Photometric settings. Blur is exposure x angular rate, so bounding the
        # exposure TIME bounds the smear regardless of how dim the room is; the
        # camera trades to gain instead, which stereo matching tolerates far
        # better than smeared edges.
        if args.exposure is not None:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            err = zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,
                                          args.exposure)
            if args.gain is not None:
                zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, args.gain)
            ms = args.exposure / 100.0 * (1000.0 / max(args.fps, 1))
            if err == sl.ERROR_CODE.SUCCESS:
                gain_now = zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)[1]
                log(f"Exposure    : LOCKED at {args.exposure}% "
                    f"(~{ms:.1f} ms at {args.fps} fps), gain {gain_now}")
            else:
                log(f"WARNING: could not lock exposure ({err}) -- auto-exposure "
                    f"stays on.")
        else:
            log("Exposure    : auto (adapts to light; can run long and smear "
                "during turns -- see --exposure)")
        if args.lock_wb is not None:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
            err = zed.set_camera_settings(
                sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, args.lock_wb)
            log(f"White bal.  : locked at {args.lock_wb} K" if
                err == sl.ERROR_CODE.SUCCESS else
                f"WARNING: could not lock white balance ({err})")

        rt = sl.RuntimeParameters()
        left_mat, right_mat, depth_mat = sl.Mat(), sl.Mat(), sl.Mat()

        preview_on = not args.no_preview
        preview_stride = 1
        if preview_on:
            res_prev = zed.get_camera_information().camera_configuration.resolution
            try:
                preview = Preview(res_prev.width, res_prev.height)
                preview_stride = max(1, round(args.fps / preview.target_hz))
                log(f"Preview     : {preview.backend} window at 1/{preview.scale} "
                    f"size, ~{args.fps / preview_stride:.0f} Hz "
                    f"(q/Esc in it stops the capture)")
            except Exception as exc:
                log(f"WARNING: preview unavailable ({exc}) -- recording "
                    f"continues headless.")
                preview_on = False

        log(f"Warm-up: {WARMUP_FRAMES} frames (auto-exposure settling) ...")
        for _ in range(WARMUP_FRAMES):
            if STOP.is_set():
                break
            if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue
            if preview_on:  # pay the window init cost before recording starts
                try:
                    if preview_status(preview, zed, left_mat, "warming up ...",
                                      (0, 220, 255)) == "stop":
                        log("Preview: q/Esc during warm-up -- aborting before "
                            "recording started.")
                        STOP.set()
                except Exception as exc:
                    log(f"WARNING: preview disabled ({exc}); recording continues.")
                    preview_on = False
        if STOP.is_set():
            log("Aborted during warm-up -- nothing recorded.")
            return 4

        # SDK 5.2.0's Python default was still the legacy GEN_1 engine (GEN_3
        # only became the global default in 5.2.1), so set everything
        # explicitly. World frame is still the pose where tracking starts.
        track_params = sl.PositionalTrackingParameters()
        track_params.mode = getattr(sl.POSITIONAL_TRACKING_MODE,
                                    args.tracking_mode.upper())
        track_params.enable_imu_fusion = True       # visual-INERTIAL, not visual-only
        track_params.enable_area_memory = True      # loop closure + relocalisation
        track_params.set_gravity_as_origin = True   # world "down" = measured gravity
        track_params.enable_2d_ground_mode = False  # docs: disable for handheld
        compute_note = ""
        pref = getattr(sl, "COMPUTE_PREFERENCE", None)          # SDK >= 5.5
        if pref is not None and hasattr(track_params, "compute_preference"):
            track_params.compute_preference = getattr(
                pref, {"gpu": "PREFER_GPU", "cpu": "PREFER_CPU",
                       "auto": "AUTO"}[args.track_compute])
            compute_note = f", compute {args.track_compute.upper()}"
        err = zed.enable_positional_tracking(track_params)
        if err != sl.ERROR_CODE.SUCCESS:
            fatal = f"enable_positional_tracking failed: {err}"
            return 2
        tracking_enabled = True
        log(f"Tracking    : {args.tracking_mode.upper()}, IMU fusion ON, "
            f"area memory (loop closure) ON{compute_note}  "
            f"[ZED SDK {sl.Camera.get_sdk_version()}]")

        # The tracker publishes nothing until it has aligned to gravity and
        # built its first keyframes: the pose sits exactly at the origin, and
        # moving the camera delays it (the SDK logs "Gravity alignment issues
        # detected"). Recording starts only once the pose is live, so poses.txt
        # never begins with placeholders and the freeze guard is valid from the
        # first recorded frame.
        log("Waiting for the tracker -- hold the camera still ...")
        live_pose, first_key, live = sl.Pose(), None, False
        t_wait = time.time()
        while (not STOP.is_set()
               and time.time() - t_wait < TRACKER_START_TIMEOUT):
            if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.get_position(live_pose, sl.REFERENCE_FRAME.WORLD)
            key = pose_key(live_pose.get_translation(sl.Translation()).get(),
                           live_pose.get_orientation(sl.Orientation()).get())
            if first_key is None:
                first_key = key
            elif key != first_key:
                live = True
                break
            if preview_on:
                try:
                    if preview_status(preview, zed, left_mat,
                                      "HOLD STILL - tracker starting ...",
                                      (0, 220, 255)) == "stop":
                        STOP.set()
                except Exception as exc:
                    log(f"WARNING: preview disabled ({exc}); recording continues.")
                    preview_on = False
        if STOP.is_set():
            log("Aborted before recording started -- nothing recorded.")
            return 4
        if not live:
            fatal = (f"tracker did not start within {TRACKER_START_TIMEOUT} s "
                     f"-- check lighting and texture, or try --tracking-mode gen1")
            return 2
        log(f"Tracker live after {time.time() - t_wait:.1f} s")

        # NVENC-backed codecs can be broken by driver/SDK mismatches (seen with
        # NVIDIA 595.79 + SDK 5.2: every H26x mode fails at encoder init while
        # ffmpeg's NVENC works). Fall back rather than lose the master copy.
        if args.no_svo:
            log("SVO2 master : DISABLED (--no-svo) -- per-frame dumps only; "
                "this session cannot be re-derived later.")
        if not args.no_svo:
            requested = args.svo_compression
            codec_chain = [requested] + [m for m in ("h265", "h264")
                                         if m != requested]
            if args.allow_lossless_svo and "lossless" not in codec_chain:
                codec_chain.append("lossless")
            svo_codec = None
            for codec in codec_chain:
                rec = sl.RecordingParameters(
                    video_filename=str(session_dir / "recording.svo2"),
                    compression_mode=SVO_COMPRESSION_MAP[codec])
                err = zed.enable_recording(rec)
                if err == sl.ERROR_CODE.SUCCESS:
                    svo_codec = codec
                    break
                log(f"WARNING: SVO2 recording with {codec.upper()} failed ({err})"
                    + (" -- trying next codec ..."
                       if codec != codec_chain[-1] else ""))
            if svo_codec is None:
                log("WARNING: no GPU-encoded SVO codec worked (NVENC is broken "
                    "on driver 595.79). NOT falling back to LOSSLESS -- at "
                    "HD1080 that is ~4 MB/frame and SVO2 files over ~4 GB "
                    "corrupt on Windows 11, so it would be unreadable anyway. "
                    "Continuing WITHOUT an SVO master; the per-frame dumps are "
                    "unaffected. Fix: roll back to a 581.x Studio driver or "
                    "update the ZED SDK, then re-run. Override with "
                    "--allow-lossless-svo.")
            else:
                recording_enabled = True
                if svo_codec != requested:
                    log(f"WARNING: SVO2 master uses {svo_codec.upper()} instead "
                        f"of {requested.upper()}. If that is LOSSLESS, keep the "
                        f"capture short -- SVO2 past ~4 GB corrupts on Win 11.")
                else:
                    log(f"SVO2 codec  : {svo_codec.upper()}")

        poses_f = open(session_dir / "poses.txt", "w", encoding="utf-8", newline="\n")
        poses_f.write("# timestamp tx ty tz qx qy qz qw\n")
        ts_f = open(session_dir / "timestamps.csv", "w", encoding="utf-8", newline="\n")
        ts_f.write("frame_index,timestamp_ns,tracking_state\n")

        writer = AsyncJpegWriter()
        globals()["_WRITER"] = writer     # so Ctrl-C can report the queue depth
        imu_f = open(session_dir / "imu.csv", "w", encoding="utf-8", newline="\n")
        imu_f.write("timestamp_ns,ax,ay,az,gx,gy,gz\n")

        expected_dt_ns = 1e9 / max(args.fps, 1)
        pose = sl.Pose()
        consecutive = 0

        log("Recording -- Ctrl-C to stop."
            + (" Press q/Esc in the preview window to stop." if preview_on else "")
            + (f" Auto-stop after {args.duration:g} s." if args.duration else ""))

        while not STOP.is_set():
            err = zed.grab(rt)
            if err != sl.ERROR_CODE.SUCCESS:
                grab_failures += 1
                consecutive += 1
                log(f"WARNING: grab failed ({err}) -- frame not recorded; "
                    f"indices stay contiguous (failure #{grab_failures})")
                if consecutive >= MAX_CONSECUTIVE_GRAB_FAILURES:
                    fatal = "too many consecutive grab failures -- camera lost?"
                    break
                time.sleep(0.005)
                continue
            consecutive = 0
            imu_last, n_imu = write_imu_batch(zed, imu_f, imu_last)
            imu_rows += n_imu

            ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            # Under load the SDK can hand back the SAME image twice (identical
            # image timestamp). Writing it would put a duplicate frame -- and a
            # non-increasing timestamp -- into the dataset, so drop it here.
            if last_ts is not None and ts_ns <= last_ts:
                dup_frames += 1
                log(f"WARNING: SDK repeated a frame (timestamp {ts_ns}) -- "
                    f"dropped, indices stay contiguous (repeat #{dup_frames})")
                continue
            if first_ts is None:
                first_ts = ts_ns
            if args.duration is not None and (ts_ns - first_ts) / 1e9 >= args.duration:
                break
            if last_ts is not None and (ts_ns - last_ts) > 1.5 * expected_dt_ns:
                gap_events += 1
                missed = round((ts_ns - last_ts) / expected_dt_ns) - 1
                log(f"WARNING: timestamp gap before frame {idx} "
                    f"(~{missed} camera frame(s) skipped by the SDK)")
            last_ts = ts_ns

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)     # rectified
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)   # rectified
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)  # float32 m, left-aligned
            writer.submit(left_dir / f"{idx:06d}.jpg",
                          left_mat.get_data(deep_copy=True))
            writer.submit(right_dir / f"{idx:06d}.jpg",
                          right_mat.get_data(deep_copy=True))
            writer.submit(depth_dir / f"{idx:06d}.png",
                          depth_mat.get_data(deep_copy=True))

            state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)  # cam-to-world
            state_name = enum_name(state)
            t = pose.get_translation(sl.Translation()).get()
            q = pose.get_orientation(sl.Orientation()).get()          # [qx,qy,qz,qw]
            if math.sqrt(float(q[0])**2 + float(q[1])**2
                         + float(q[2])**2 + float(q[3])**2) < 1e-6:
                q = (0.0, 0.0, 0.0, 1.0)  # tracking not initialized yet

            key = pose_key(t, q)
            if key == prev_pose_key:
                frozen_run += 1
                if frozen_run == 30:                 # ~2 s at 15 fps
                    freeze_events += 1
                    log(f"WARNING: POSE FROZEN -- the tracker has not updated "
                        f"for 2 s (from frame {idx - 30}) while claiming "
                        f"{state_name}. Poses from here are invalid. Stop and "
                        f"restart the capture.")
                if frozen_run >= 30:
                    frozen_frames += 1
                if frozen_run == 45 and not args.no_freeze_stop:
                    log("STOPPING: tracker frozen for 3 s. Everything recorded "
                        "before the freeze is kept -- start a new capture.")
                    STOP.set()
            else:
                frozen_run = 0
            prev_pose_key = key

            ts_str = f"{ts_ns // 10**9}.{ts_ns % 10**9:09d}"  # exact ns, no float loss
            poses_f.write(f"{ts_str} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                          f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
            ts_f.write(f"{idx},{ts_ns},{state_name}\n")
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
            if state_name == "OK":
                ok_frames += 1

            try:
                st = zed.get_positional_tracking_status()
                mem = enum_name(st.spatial_memory_status)
                odo = enum_name(st.odometry_status)
            except Exception:           # diagnostics only -- never fatal
                st, mem, odo = None, "?", "?"
            if odo == "INSUFFICIENT_FEATURES":
                insufficient_feat += 1
            if mem == "LOOP_CLOSED" and last_mem != "LOOP_CLOSED":
                loop_closures += 1
                loop_flash = 30                     # ~2 s on the HUD
                log(f"LOOP CLOSED #{loop_closures} at frame {idx} "
                    f"(t={(ts_ns - first_ts) / 1e9:.1f}s)")
            last_mem = mem
            loop_flash = max(0, loop_flash - 1)
            if idx == 30 and st is not None:  # settled; confirm IMU is fused
                fusion_name = enum_name(st.tracking_fusion_status)
                log(f"Fusion      : {fusion_name}"
                    + ("" if "INERTIAL" in fusion_name else
                       "  <-- WARNING: IMU is NOT being fused"))

            if preview_on and idx % preview_stride == 0:  # ~15 Hz refresh
                try:
                    view = left_mat.get_data(deep_copy=False)
                    small = cv2.cvtColor(
                        cv2.resize(view, (view.shape[1] // preview.scale,
                                          view.shape[0] // preview.scale)),
                        cv2.COLOR_BGRA2BGR)
                    elapsed = (ts_ns - first_ts) / 1e9
                    hud = (f"REC {elapsed:5.0f}s  f{idx}  "
                           f"fps~{(idx / elapsed) if elapsed > 0 else 0.0:4.1f}  "
                           f"{state_name}  loops {loop_closures}"
                           + ("  << LOOP CLOSED" if loop_flash else ""))
                    if frozen_run >= 30:
                        hud = f"!! POSE FROZEN {frozen_run / 15:.0f}s -- STOP !!"
                    color = ((0, 255, 0) if state_name == "OK"
                             and frozen_run < 30 else (0, 0, 255))
                    cv2.putText(small, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 1, cv2.LINE_AA)
                    result = preview.show(small)
                    if result == "stop":
                        log("Preview: q/Esc pressed -- stopping capture ...")
                        STOP.set()
                    elif result == "closed":
                        log("Preview window closed -- recording continues "
                            "(Ctrl-C to stop).")
                        preview.close()
                        preview_on = False
                except Exception as exc:
                    log(f"WARNING: preview disabled ({exc}); recording continues.")
                    preview_on = False

            idx += 1
            if idx % 75 == 0:  # status every ~5 s at 15 fps
                poses_f.flush()
                ts_f.flush()
                imu_f.flush()
                elapsed = (ts_ns - first_ts) / 1e9
                fps_now = (idx - 1) / elapsed if elapsed > 0 else 0.0
                log(f"frame {idx:6d} | t={elapsed:7.1f}s | fps~{fps_now:4.1f} "
                    f"| tracking={state_name} odo={odo} mem={mem} "
                    f"loops={loop_closures} "
                    f"| ok={100.0 * ok_frames / idx:5.1f}% "
                    f"| write-queue={writer.qsize()} | grab-fails={grab_failures}")

    except KeyboardInterrupt:
        log("KeyboardInterrupt -- stopping.")
    finally:
        if preview is not None:   # tear the GUI down first; Tcl dislikes late teardown
            preview.close()
        if imu_f is not None:
            imu_f.close()
        if writer is not None:
            log("Finalizing: draining image write queue ...")
            writer.close()
        for f in (poses_f, ts_f):
            if f is not None:
                f.flush()
                f.close()
        if recording_enabled:
            zed.disable_recording()
        if tracking_enabled:
            # GEN_3 keyframes: the frames its pose graph is built on. Whether
            # their poses are re-optimised after loop closure is undocumented,
            # so they are exported alongside poses.txt for comparison, never
            # instead of it.
            try:
                kfs = {}
                if (zed.get_positional_tracking_keyframes(kfs)
                        == sl.ERROR_CODE.SUCCESS and kfs):
                    rows = []
                    for kf in kfs.values():
                        kt = kf.timestamp.get_nanoseconds()
                        tr = kf.pose.get_translation().get()
                        qo = kf.pose.get_orientation().get()
                        rows.append((kt, tr, qo))
                    rows.sort(key=lambda r: r[0])
                    with open(session_dir / "keyframes.txt", "w",
                              encoding="utf-8", newline="\n") as kf_f:
                        kf_f.write("# timestamp tx ty tz qx qy qz qw  "
                                   "(GEN_3 keyframes, same frame as poses.txt)\n")
                        for kt, tr, qo in rows:
                            kf_f.write(f"{kt // 10**9}.{kt % 10**9:09d} "
                                       f"{tr[0]:.6f} {tr[1]:.6f} {tr[2]:.6f} "
                                       f"{qo[0]:.9f} {qo[1]:.9f} {qo[2]:.9f} "
                                       f"{qo[3]:.9f}\n")
                    log(f"keyframes   : {len(rows)} exported to keyframes.txt")
            except Exception as exc:
                log(f"WARNING: keyframe export failed ({exc})")
            if not args.no_save_area and idx > 0:
                try:
                    export_area_map(zed, session_dir / "area.area")
                except Exception as exc:
                    log(f"WARNING: area export failed ({exc}) -- dataset "
                        f"unaffected")
            zed.disable_positional_tracking()
        zed.close()

        duration = (last_ts - first_ts) / 1e9 if (first_ts is not None
                                                  and last_ts is not None
                                                  and idx > 1) else 0.0
        svo = session_dir / "recording.svo2"
        svo_note = (f"{svo.stat().st_size / 1e6:.1f} MB" if svo.exists()
                    else "none (--no-svo)" if args.no_svo else "MISSING")
        expected_files = 3 * idx  # left + right JPEG, depth PNG
        log("---- capture summary ----")
        log(f"frames grabbed       : {idx}")
        if duration > 0:
            achieved_fps = (idx - 1) / duration
            log(f"duration             : {duration:.1f} s")
            log(f"achieved fps         : {achieved_fps:.2f} (target {args.fps})")
            if achieved_fps < 0.87 * args.fps:
                log("WARNING: sustained rate is below target (~13 fps floor at "
                    "HD720@15). Reported honestly: every dumped frame is still a "
                    "consistent L/R/depth triple; SDK frame gaps are logged above. "
                    "Nothing was silently dropped.")
        log(f"grab failures        : {grab_failures}")
        log(f"SDK frame-gap events : {gap_events}")
        log(f"repeated frames dropped : {dup_frames}")
        log(f"tracking engine      : {args.tracking_mode.upper()}  "
            f"(fusion {fusion_name})")
        log(f"loop closures        : {loop_closures}")
        if freeze_events:
            log(f"POSE FROZEN          : {freeze_events} episode(s), "
                f"{frozen_frames} frames -- poses.txt is INVALID there")
        if idx and insufficient_feat:
            log(f"low-feature frames   : {insufficient_feat} "
                f"({100.0 * insufficient_feat / idx:.1f}%) -- point at more "
                f"texture, less blank wall/floor")
        log(f"tracking states      : {state_counts}")
        if idx:
            pct = 100.0 * ok_frames / idx
            log(f"tracking OK          : {pct:.1f}%"
                + ("  (LOW -- consider re-recording)" if pct < 80 else ""))
        log(f"imu rows             : {imu_rows}"
            + (f" (~{imu_rows / duration:.0f} Hz)" if duration > 0 else ""))
        jw = writer.written if writer else 0
        jf = len(writer.failed) if writer else 0
        log(f"files written        : {jw} of {expected_files} expected "
            f"(2 JPEG + 1 depth PNG per frame)"
            + (f"  ({jf} FAILED)" if jf else ""))
        log(f"svo2 master          : {svo_note}")
        log(f"session dir          : {session_dir.resolve()}")
        if fatal:
            log(f"FATAL: {fatal}")
        log(f'Next: python scripts/zed_verify_session.py "{session_dir}"')
        if _LOGFILE is not None:
            _LOGFILE.close()

    if idx > 0 and not args.no_verify:
        print("\n===== verifying the session =====", flush=True)
        subprocess.run([sys.executable,
                        str(Path(__file__).with_name("zed_verify_session.py")),
                        str(session_dir)])

    if fatal:
        return 3
    if idx == 0:
        print("ERROR: no frames captured.", file=sys.stderr)
        return 4
    if writer is not None and (writer.failed or writer.written != expected_files):
        print("ERROR: image writer did not persist every frame -- session is "
              "incomplete; run the verifier for details.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
