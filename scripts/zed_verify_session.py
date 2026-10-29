#!/usr/bin/env python3
"""
zed_verify_session.py -- validate a zed_session_* directory before shipping it
to the compute server.

Hard-fails (exit 1, clear message) unless:
  * images/left and images/right have equal counts, 6-digit contiguous indices,
    and match the timestamps.csv row count,
  * depth/ has one uint16 PNG per left frame (same indices); on 5 sampled
    frames the valid-pixel fraction is > 50% and the median depth is within
    [0.3, 10] m,
  * poses.txt has the TUM header, one line per frame, unit quaternions, and
    no frozen-tracker stretch (>= 30 identical consecutive poses),
  * camera_info.json parses with fx,fy,cx,cy > 0, baseline_m in [0.10, 0.14],
    depth_units == "mm" and depth_aligned_to == "left",
  * timestamps are strictly increasing.
Reports the tracking_state==OK fraction (warns below 80%) and prints a summary:
n_frames, duration, distance traveled (from poses), mean speed, depth stats.

Runs on plain Python (no pyzed needed); OpenCV + numpy are required for the
depth checks and the JPEG decode spot-check.

Usage:
  python scripts/zed_verify_session.py captures/zed_session_YYYYMMDD_HHMMSS
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None

POSES_HEADER = "# timestamp tx ty tz qx qy qz qw"
TIMESTAMPS_HEADER = "frame_index,timestamp_ns,tracking_state"
IMU_HEADER = "timestamp_ns,ax,ay,az,gx,gy,gz"
BASELINE_RANGE = (0.10, 0.14)   # meters; ZED 2 / ZED 2i nominal 0.12
QNORM_TOL = 1e-2
OK_WARN_FRACTION = 0.80

FAILS, WARNS = [], []


def ok(msg):   print(f"[ ok ] {msg}")
def info(msg): print(f"[info] {msg}")
def warn(msg): WARNS.append(msg); print(f"[WARN] {msg}")
def fail(msg): FAILS.append(msg); print(f"[FAIL] {msg}")


def bail():
    print(f"\nVERIFICATION FAILED ({len(FAILS)} hard failure(s)):")
    for m in FAILS:
        print(f"  - {m}")
    sys.exit(1)


def check_frame_dir(d: Path, label: str, ext: str) -> int:
    files = sorted(d.glob(f"*.{ext}"))
    n = len(files)
    bad = [f.name for f in files if not (len(f.stem) == 6 and f.stem.isdigit())]
    if bad:
        fail(f"{label}: {len(bad)} file(s) not 6-digit zero-padded "
             f"(first: {bad[0]})")
        return n
    for i, f in enumerate(files):
        if int(f.stem) != i:
            fail(f"{label}: indices not contiguous -- expected "
                 f"{i:06d}.{ext}, found {f.name}")
            return n
    ok(f"{label}: {n} frames, contiguous 000000..{n - 1:06d}")
    return n


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if len(s) > 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("session_dir", help="path to a zed_session_* directory")
    ap.add_argument("--rsync-target",
                    default="<user>@<server>:<datasets_dir>/",
                    help="shown in the hand-off command printed on success")
    args = ap.parse_args()

    s = Path(args.session_dir).expanduser()
    print(f"Verifying session: {s}\n")
    if not s.is_dir():
        fail(f"not a directory: {s}")
        bail()

    # ---- layout ---------------------------------------------------------
    required = ["camera_info.json", "poses.txt", "timestamps.csv", "imu.csv",
                "images/left", "images/right", "depth"]
    missing = [r for r in required if not (s / r).exists()]
    for r in missing:
        fail(f"missing required entry: {r}")
    if missing:
        bail()
    ok("layout: all required files/dirs present")
    if not (s / "recording.svo2").exists():
        warn("recording.svo2 not present (master copy missing -- dumps only)")

    # ---- images + depth --------------------------------------------------
    n_left = check_frame_dir(s / "images" / "left", "images/left", "jpg")
    n_right = check_frame_dir(s / "images" / "right", "images/right", "jpg")
    if n_left != n_right:
        fail(f"left/right image counts differ: {n_left} vs {n_right}")
    if n_left == 0:
        fail("no frames in images/left")
        bail()
    n_frames = n_left
    n_depth = check_frame_dir(s / "depth", "depth", "png")
    if n_depth != n_frames:
        fail(f"depth count {n_depth} != left image count {n_frames}")

    # ---- timestamps.csv -------------------------------------------------
    ts_lines = (s / "timestamps.csv").read_text(encoding="utf-8").splitlines()
    if not ts_lines or ts_lines[0].strip() != TIMESTAMPS_HEADER:
        fail(f"timestamps.csv header must be exactly '{TIMESTAMPS_HEADER}' "
             f"(got: {ts_lines[0].strip() if ts_lines else '<empty>'!r})")
        bail()
    ts_ns, states = [], []
    for lineno, line in enumerate(ts_lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 3:
            fail(f"timestamps.csv line {lineno}: expected 3 columns, got "
                 f"{len(parts)}")
            bail()
        try:
            i, t = int(parts[0]), int(parts[1])
        except ValueError:
            fail(f"timestamps.csv line {lineno}: non-integer index/timestamp")
            bail()
        if i != len(ts_ns):
            fail(f"timestamps.csv line {lineno}: frame_index {i} != row "
                 f"position {len(ts_ns)} (must be sequential from 0)")
            bail()
        ts_ns.append(t)
        states.append(parts[2].strip())
    if len(ts_ns) != n_frames:
        fail(f"timestamps.csv has {len(ts_ns)} rows but images have "
             f"{n_frames} frames")
    else:
        ok(f"timestamps.csv: {len(ts_ns)} rows == image count")

    nonmono = sum(1 for a, b in zip(ts_ns, ts_ns[1:]) if b <= a)
    if nonmono:
        fail(f"timestamps.csv: {nonmono} non-increasing timestamp step(s)")
    duration = (ts_ns[-1] - ts_ns[0]) / 1e9 if len(ts_ns) > 1 else 0.0
    if len(ts_ns) > 1:
        dts = sorted(b - a for a, b in zip(ts_ns, ts_ns[1:]))
        median_dt = dts[len(dts) // 2]
        gaps = sum(1 for d in dts if d > 1.5 * median_dt)
        est_fps = 1e9 / median_dt if median_dt > 0 else 0.0
        if gaps:
            warn(f"{gaps} inter-frame gap(s) > 1.5x median dt "
                 f"(frames skipped during capture)")
    else:
        est_fps = 0.0

    # ---- poses.txt ------------------------------------------------------
    pose_lines = (s / "poses.txt").read_text(encoding="utf-8").splitlines()
    if not pose_lines or pose_lines[0].strip() != POSES_HEADER:
        fail(f"poses.txt header must be exactly '{POSES_HEADER}'")
        bail()
    data_lines = [l for l in pose_lines[1:] if l.strip()]
    if len(data_lines) != n_frames:
        fail(f"poses.txt has {len(data_lines)} pose lines but images have "
             f"{n_frames} frames")
    poses = []
    max_qdev = 0.0
    bad_q = 0
    for lineno, line in enumerate(data_lines, start=2):
        vals = line.split()
        if len(vals) != 8:
            fail(f"poses.txt line {lineno}: expected 8 values, got {len(vals)}")
            bail()
        try:
            t, tx, ty, tz, qx, qy, qz, qw = (float(v) for v in vals)
        except ValueError:
            fail(f"poses.txt line {lineno}: non-numeric value")
            bail()
        qn = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        max_qdev = max(max_qdev, abs(qn - 1.0))
        if abs(qn - 1.0) > QNORM_TOL:
            bad_q += 1
        poses.append((t, tx, ty, tz))
    if bad_q:
        fail(f"poses.txt: {bad_q} quaternion(s) not normalized "
             f"(max | |q|-1 | = {max_qdev:.4f}, tol {QNORM_TOL})")
    else:
        ok(f"poses.txt: {len(data_lines)} lines, quaternions normalized "
           f"(max | |q|-1 | = {max_qdev:.2e})")

    # Frozen tracker: identical consecutive poses. A hand-held camera never
    # holds a bit-identical pose, so a long run means the tracker stopped
    # updating even if tracking_state still says OK.
    run = longest = 0
    longest_end = 0
    full = [tuple(round(float(v), 7) for v in l.split()[1:8]) for l in data_lines]
    for i in range(1, len(full)):
        run = run + 1 if full[i] == full[i - 1] else 0
        if run > longest:
            longest, longest_end = run, i
    if longest >= 30:
        fps_guess = (len(ts_ns) - 1) / ((ts_ns[-1] - ts_ns[0]) / 1e9) \
            if len(ts_ns) > 1 and ts_ns[-1] > ts_ns[0] else 15.0
        fail(f"poses.txt: tracker FROZE -- {longest + 1} identical consecutive "
             f"poses (~{(longest + 1) / fps_guess:.0f} s, frames "
             f"{longest_end - longest}..{longest_end}); poses there are invalid")
    else:
        ok(f"poses.txt: no frozen-tracker stretch (longest identical run "
           f"{longest + 1} frame(s))")

    if poses and len(poses) == len(ts_ns):
        max_dt = max(abs(p[0] - t / 1e9) for p, t in zip(poses, ts_ns))
        if max_dt > 2e-3:
            warn(f"poses.txt timestamps deviate from timestamps.csv by up to "
                 f"{max_dt * 1e3:.2f} ms")

    # ---- camera_info.json ----------------------------------------------
    try:
        with open(s / "camera_info.json", encoding="utf-8") as f:
            ci = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        fail(f"camera_info.json unreadable: {exc}")
        bail()
    for side in ("left_camera", "right_camera"):
        cam = ci.get(side)
        if not isinstance(cam, dict):
            fail(f"camera_info.json: missing {side}")
            continue
        for k in ("fx", "fy", "cx", "cy"):
            v = cam.get(k)
            if not isinstance(v, (int, float)) or not v > 0:
                fail(f"camera_info.json: {side}.{k} must be > 0 (got {v!r})")
    baseline = ci.get("baseline_m")
    if not isinstance(baseline, (int, float)) or not (
            BASELINE_RANGE[0] <= baseline <= BASELINE_RANGE[1]):
        fail(f"camera_info.json: baseline_m {baseline!r} outside "
             f"[{BASELINE_RANGE[0]}, {BASELINE_RANGE[1]}] m -- wrong unit or "
             f"wrong camera model?")
    else:
        ok(f"camera_info.json: intrinsics sane, baseline {baseline:.4f} m "
           f"({ci.get('model', '?')}, SN {ci.get('serial_number', '?')})")
    dm = ci.get("depth_mode")
    if not (isinstance(dm, str) and dm):
        fail(f"camera_info.json: depth_mode missing/empty (got {dm!r})")
    if ci.get("depth_units") != "mm":
        fail(f"camera_info.json: depth_units must be 'mm' "
             f"(got {ci.get('depth_units')!r})")
    if ci.get("depth_aligned_to") != "left":
        fail(f"camera_info.json: depth_aligned_to must be 'left' "
             f"(got {ci.get('depth_aligned_to')!r})")

    # ---- sample-decode spot check ---------------------------------------
    res = ci.get("resolution", {})
    w, h = res.get("width"), res.get("height")
    if cv2 is not None and isinstance(w, int) and isinstance(h, int):
        sample_ids = sorted({0, n_frames // 2, n_frames - 1})
        bad = 0
        for side in ("left", "right"):
            for i in sample_ids:
                p = s / "images" / side / f"{i:06d}.jpg"
                img = cv2.imread(str(p))
                if img is None:
                    fail(f"sample decode: {p.name} ({side}) unreadable/corrupt")
                    bad += 1
                elif img.shape[1] != w or img.shape[0] != h:
                    fail(f"sample decode: {p.name} ({side}) is "
                         f"{img.shape[1]}x{img.shape[0]}, expected {w}x{h}")
                    bad += 1
        if not bad:
            ok(f"sample decode: {2 * len(sample_ids)} sample JPEGs decode at "
               f"{w}x{h}")
    else:
        warn("OpenCV not available or resolution missing -- skipped JPEG "
             "decode spot-check")

    # ---- depth maps (5 sampled frames) ------------------------------------
    depth_fracs, depth_medians = [], []
    if cv2 is None or np is None:
        fail("depth checks require opencv + numpy in the verification env")
    else:
        depth_ok = True
        ids = sorted({round(i * (n_frames - 1) / 4) for i in range(5)})
        for i in ids:
            p = s / "depth" / f"{i:06d}.png"
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                fail(f"depth: {p.name} unreadable/corrupt")
                depth_ok = False
                continue
            if img.dtype != np.uint16 or img.ndim != 2:
                fail(f"depth: {p.name} must be single-channel uint16 "
                     f"(got dtype={img.dtype}, ndim={img.ndim})")
                depth_ok = False
                continue
            if isinstance(w, int) and isinstance(h, int) and img.shape != (h, w):
                fail(f"depth: {p.name} is {img.shape[1]}x{img.shape[0]}, "
                     f"expected {w}x{h}")
                depth_ok = False
                continue
            valid_frac = float((img > 0).mean())
            depth_fracs.append(valid_frac)
            if valid_frac <= 0.50:
                fail(f"depth: {p.name} valid-pixel fraction "
                     f"{100 * valid_frac:.1f}% <= 50% (expected >50% indoors)")
                depth_ok = False
                continue
            med_m = float(np.median(img[img > 0])) / 1000.0
            depth_medians.append(med_m)
            if not (0.3 <= med_m <= 10.0):
                fail(f"depth: {p.name} median depth {med_m:.2f} m outside "
                     f"[0.3, 10] m")
                depth_ok = False
        if depth_ok and depth_fracs:
            ok(f"depth: {len(ids)} samples uint16 mm, valid "
               f"{100 * min(depth_fracs):.0f}-{100 * max(depth_fracs):.0f}%, "
               f"median {min(depth_medians):.2f}-{max(depth_medians):.2f} m")

    # ---- imu.csv ---------------------------------------------------------
    imu_lines = (s / "imu.csv").read_text(encoding="utf-8").splitlines()
    n_imu = max(0, len([l for l in imu_lines[1:] if l.strip()]))
    if not imu_lines or imu_lines[0].strip() != IMU_HEADER:
        warn(f"imu.csv header differs from '{IMU_HEADER}'")
    if duration > 0 and n_imu < 50 * duration:
        warn(f"imu.csv: only {n_imu} rows over {duration:.1f} s "
             f"(~{n_imu / duration:.0f} Hz -- expected several hundred Hz)")

    # ---- tracking quality ------------------------------------------------
    n_ok = sum(1 for st in states if st == "OK")
    frac = n_ok / len(states) if states else 0.0
    line = (f"tracking_state OK: {n_ok}/{len(states)} frames "
            f"({100 * frac:.1f}%)")
    if frac < OK_WARN_FRACTION:
        warn(line + f"  -- below {int(OK_WARN_FRACTION * 100)}%, consider "
             f"re-recording")
    else:
        ok(line)

    # ---- trajectory summary ----------------------------------------------
    dist = 0.0
    max_step = 0.0
    for i in range(1, min(len(poses), len(states))):
        if states[i] == "OK" and states[i - 1] == "OK":
            dx = poses[i][1] - poses[i - 1][1]
            dy = poses[i][2] - poses[i - 1][2]
            dz = poses[i][3] - poses[i - 1][3]
            step = math.sqrt(dx * dx + dy * dy + dz * dz)
            dist += step
            max_step = max(max_step, step)
    if max_step > 0.5:
        warn(f"largest single-frame jump is {max_step:.2f} m -- possible "
             f"tracking glitch")
    mean_speed = dist / duration if duration > 0 else 0.0
    if mean_speed > 0.6:
        warn(f"mean speed {mean_speed:.2f} m/s exceeds the <0.5 m/s protocol")
    end_offset = 0.0
    if len(poses) > 1:
        dx = poses[-1][1] - poses[0][1]
        dy = poses[-1][2] - poses[0][2]
        dz = poses[-1][3] - poses[0][3]
        end_offset = math.sqrt(dx * dx + dy * dy + dz * dz)
    size_gb = sum(p.stat().st_size for p in s.rglob("*") if p.is_file()) / 1e9

    print("\n---- session summary ----")
    print(f"n_frames          : {n_frames}")
    print(f"duration          : {duration:.1f} s  (~{est_fps:.1f} fps)")
    print(f"distance traveled : {dist:.2f} m   (over tracking-OK segments)")
    print(f"mean speed        : {mean_speed:.2f} m/s")
    print(f"start->end offset : {end_offset:.2f} m   (small => loop closed)")
    print(f"tracking OK       : {100 * frac:.1f}%")
    if depth_fracs:
        print(f"depth (samples)   : valid {100 * sum(depth_fracs) / len(depth_fracs):.1f}%"
              + (f", median {sorted(depth_medians)[len(depth_medians) // 2]:.2f} m"
                 if depth_medians else ""))
    print(f"size on disk      : {size_gb:.2f} GB")
    if WARNS:
        print(f"warnings          : {len(WARNS)}")

    if FAILS:
        bail()

    print(f"\nSESSION VERIFIED: {s.name}")
    print("\nShip it to the compute server (rsync from WSL preserves the layout):")
    print(f'  wsl rsync -avP "{wsl_path(s)}/" {args.rsync_target}{s.name}/')
    print("  (append --exclude recording.svo2 --exclude capture.log to ship "
          "only the pipeline payload)")
    print("Plain-Windows alternative (scp copies the folder into the target dir):")
    print(f'  scp -r "{s.resolve()}" {args.rsync_target}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
