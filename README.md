ACLab: Tello Wall Corridor Follower (No Markers)
================================================

This project controls a DJI Tello to follow the middle red band on a wall using a fixed webcam. No markers are attached to the drone. Vision runs offboard and drives velocity commands over Wi‑Fi.

Quick start
-----------
1) Install Python 3.9+ and pip, then:

```bash
pip install -r requirements.txt
```

2) Calibrate the webcam (intrinsics):
```bash
python -m calibration.calibrate_camera --camera 0 --out calibration/config.yaml
```

3) Optional: calibrate wall homography (click 4 wall points):
```bash
python -m calibration.calibrate_wall_homography --camera 0 --config calibration/config.yaml
```

4) Dry‑run vision only (no Tello commands):
```bash
python -m scripts.offline_test --camera 0 --config calibration/config.yaml
```

5) Live run (connect PC to Tello Wi‑Fi first):
```bash
python -m main --camera 0 --config calibration/config.yaml
```

Operational notes
-----------------
- Place the Tello on the ground near the right corner of the middle band. After takeoff the system will home to an entry waypoint near the bottom‑right corner, align to CCW, then begin the lap.
- Emergency keys: SPACE to land, Q to immediate stop/land.
- Tune HSV thresholds in `config.yaml` if lighting changes.

Repository layout
-----------------
- `main.py` — entrypoint with the homing/following state machine
- `vision/route_detector.py` — red band detection and centerline extraction
- `vision/drone_tracker.py` — markerless tracker (band-masked motion + dark blob + Kalman)
- `control/tello_controller.py` — Tello wrapper (with dry-run)
- `control/wall_follow.py` — homing and corridor following controller
- `calibration/*.py` — camera and optional wall homography calibration
- `utils/*` — configuration and geometry helpers


