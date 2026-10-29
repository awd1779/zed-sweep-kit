# ZED sweep kit

Records a mapping sweep of a floor with a ZED 2 / ZED 2i: stereo images, depth,
camera poses (with the ZED's loop closure) and IMU, plus the `.svo2` recording.
Windows; an NVIDIA GPU is needed.

```
zed_sweep_kit/
├── capture.bat                    double-click to record (+ auto-verify)
├── requirements.txt
└── scripts/
    ├── zed_capture_session.py     the recorder (live preview window)
    └── zed_verify_session.py      checks a finished session
```

## 1. Install (once)

1. **ZED SDK 5.5** (or newer 5.x) for your CUDA version:
   https://www.stereolabs.com/developers/release — reboot when it asks.
2. **ZED Python API** (`pyzed`, not on pip). From the Python you will use:
   ```
   python "C:\Program Files (x86)\ZED SDK\get_python_api.py"
   ```
3. The other packages:
   ```
   python -m pip install -r requirements.txt
   ```
4. Plug the camera **directly into a USB 3 port** (no hub). If anything fails to
   open, run `C:\Program Files (x86)\ZED SDK\tools\ZED Diagnostic.exe`, and apply
   a firmware update if it offers one.

The first run downloads and optimises the NEURAL depth model, which can take a
few minutes. Do a 20-second test first:

```
capture.bat --duration 20
```

## 2. Record

Double-click **`capture.bat`**. It records at 1080p, 15 fps, with white balance
locked at 5600 K (stops the yellow tint and colour flicker), into
`captures\zed_session_<date_time>\`.

- **Hold the camera still** until the preview changes from
  `HOLD STILL - tracker starting` to `REC` (under a second).
- Stop with **q / Esc** in the preview window, or Ctrl-C once.
- When it stops, the session is verified automatically. `SESSION VERIFIED` means
  it's good. Otherwise re-record while you're still on site.

Exposure is automatic. If the space has even lighting and you see motion blur,
you can lock it: `capture.bat --exposure 15 --gain 30` (adjust the gain until
the image isn't dark).

## 3. How to walk (this decides map quality)

The ZED's tracker (GEN_3) **closes loops live**: when you come back to a place
it has seen, it corrects its drift. The preview shows `loops N` and flashes
`LOOP CLOSED`. To give it loops:

- **Walk circuits of at most ~20 m**, not out-and-back. It recognises a place
  only if you pass it **facing the same way** as before.
- **Revisit covered ground every 60–90 s.** If `loops` hasn't gone up in a
  couple of minutes, go back somewhere you've been.
- **0.3–0.4 m/s**, a slow walk. Translate while turning; don't spin on the spot.
- Camera at **1.2–1.6 m**, held **level** (not aimed at the floor), steady grip.
- Stay **0.3–10 m** from surfaces. Avoid bare white walls, glass and strong
  backlight.
- **Finish where you started** and hold still for a few seconds.

Note: loop closure corrects poses **from that moment on**. Poses already written
to `poses.txt` before a loop closed keep their drift. The tracker also misses
some revisits, so walking short circuits matters.

If the preview turns red with `POSE FROZEN`, the tracker stalled, and the
capture stops by itself. Everything before that is kept; just start again.

## 4. What you get

```
zed_session_<YYYYMMDD_HHMMSS>/
├── recording.svo2     full recording; replay it in ZED Explorer / Depth Viewer
│                      or with the SDK (InitParameters.set_from_svo_file)
├── area.area          ZED spatial-memory map (relocalise later sessions into
│                      this one's coordinates)
├── camera_info.json   rectified intrinsics, baseline, model/serial, depth info
├── images/left/       000000.jpg ... rectified left frames
├── images/right/      same indices, rectified right frames
├── depth/             000000.png ... uint16 depth in mm, aligned to left, 0 = invalid
├── poses.txt          TUM: timestamp tx ty tz qx qy qz qw (camera-to-world, metres)
├── imu.csv            timestamp_ns, accel (m/s², gravity incl.), gyro (deg/s)
├── timestamps.csv     frame_index, timestamp_ns, tracking_state
├── keyframes.txt      tracker keyframe poses
└── capture.log        warnings and summary
```

Poses use the OpenCV camera frame (x right, y down, z forward), with the origin
where recording started. Disk use is roughly 2 GB per minute (mostly depth
PNGs); the `.svo2` alone is about 110 MB per minute.

To re-check a session by hand:

```
python scripts\zed_verify_session.py captures\zed_session_<date_time>
```

All options: `python scripts\zed_capture_session.py --help`
