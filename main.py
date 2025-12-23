from __future__ import annotations

import argparse
import sys
import time
import cv2
import numpy as np

from vision.open_route_detector import OpenRouteDetector
from utils.config import load_config
from control.tello_controller import TelloController
from control.wall_follow import WallFollower, State
from control.laser_follow import run_laser_follow
from vision.route_detector import RouteDetector
from vision.open_route_detector import OpenRouteDetector
from vision.drone_tracker import DroneTracker
from vision.laser_detector import LaserDetector
from utils.geometry import generate_checkpoints, generate_checkpoints_open


def draw_overlays(frame, det, drone_xy, checkpoints=None, current_checkpoint_idx=None, blue_mask=None, open_route: bool = False):
	if det is not None:
		outer = det.get("outer")
		inner = det.get("inner")
		centerline = det.get("centerline")
		if not open_route:
			if outer is not None:
				cv2.polylines(frame, [outer.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
			if inner is not None:
				cv2.polylines(frame, [inner.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
		# Always draw centerline if available (works for open route)
		if centerline is not None:
			cv2.polylines(frame, [centerline.astype(int).reshape(-1, 1, 2)], False, (255, 0, 0), 2)

	# Draw blue detection mask overlay (semi-transparent cyan)
	if blue_mask is not None:
		# Create colored overlay: cyan for detected blue areas
		blue_overlay = frame.copy()
		blue_overlay[blue_mask > 0] = [255, 255, 0]  # Cyan for blue detection
		cv2.addWeighted(frame, 0.7, blue_overlay, 0.3, 0, frame)

	# Draw checkpoints if provided
	if checkpoints is not None and len(checkpoints) > 0:
		checkpoints_int = checkpoints.astype(int)
		# Draw lines connecting checkpoints (projected path)
		for i in range(len(checkpoints_int)):
			if open_route and i == len(checkpoints_int) - 1:
				# Do not wrap for open routes
				break
			next_idx = i + 1 if open_route else (i + 1) % len(checkpoints_int)
			cv2.line(frame, 
			         tuple(checkpoints_int[i]), 
			         tuple(checkpoints_int[next_idx]), 
			         (255, 255, 0), 2)  # Yellow path
		
		# Draw checkpoints as circles
		# Current checkpoint: 8px radius (cyan, filled) - this is the upcoming direction point
		# Other checkpoints: 5px radius (yellow, outline only)
		for i, cp in enumerate(checkpoints_int):
			color = (0, 255, 255) if i == current_checkpoint_idx else (255, 255, 0)
			thickness = -1 if i == current_checkpoint_idx else 2
			radius = 8 if i == current_checkpoint_idx else 5
			cv2.circle(frame, tuple(cp), radius, color, thickness)
			# Draw checkpoint number (1-indexed: 1, 2, ..., 30)
			checkpoint_num = i + 1
			cv2.putText(frame, str(checkpoint_num), (cp[0] + 10, cp[1]), 
			           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

	if drone_xy is not None:
		# Drone detection point: 6px radius (cyan, filled) + 10px outer ring (cyan outline) - reduced size
		cv2.circle(frame, tuple(map(int, drone_xy)), 6, (0, 255, 255), -1)  # Cyan dot for drone (reduced from 8px)
		cv2.circle(frame, tuple(map(int, drone_xy)), 10, (0, 255, 255), 2)  # Outer ring (reduced from 12px)
	else:
		# Show warning text if drone not detected
		cv2.putText(frame, "NO TARGET DETECTED", (10, frame.shape[0] - 20),
		           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def find_logitech_streamcam() -> int:
	"""
	Automatically find the Logitech StreamCam by scanning available cameras.
	Returns the camera index, or -1 if not found.
	
	Strategy: StreamCam is usually not index 0 (default webcam), so we try
	indices 1, 2, 3 first, then fall back to 0.
	"""
	import platform
	import warnings
	
	# Suppress OpenCV warnings during detection
	warnings.filterwarnings('ignore', category=UserWarning)
	
	# Try backends in order: MSMF (Media Foundation) is often more reliable on Windows
	backends = []
	if hasattr(cv2, "CAP_MSMF"):
		backends.append(cv2.CAP_MSMF)  # Try MSMF first (more reliable on Windows 10+)
	if hasattr(cv2, "CAP_DSHOW"):
		backends.append(cv2.CAP_DSHOW)  # Then DirectShow
	backends.append(cv2.CAP_ANY)  # Finally, any available backend
	
	# Try indices in order: StreamCam is often 1 or 2, not 0
	# Priority: 1, 2, 3 (common StreamCam indices), then 0, then others
	candidate_indices = [1, 2, 3, 0, 4, 5, 6, 7, 8, 9]
	
	working_cameras = []  # Store all working cameras
	
	for idx in candidate_indices:
		for backend in backends:
			try:
				cap = cv2.VideoCapture(idx, backend)
				if cap.isOpened():
					ret, frm = cap.read()
					if ret and frm is not None and frm.size and float(frm.mean()) > 1.0:
						# Found a working camera
						cap.release()
						backend_name = "MSMF" if backend == cv2.CAP_MSMF else ("DSHOW" if backend == cv2.CAP_DSHOW else "ANY")
						print(f"  → Camera {idx} (backend: {backend_name}): Working ✓")
						working_cameras.append((idx, backend))
						# Prefer indices 1-3 for StreamCam (usually not 0)
						if idx in [1, 2, 3]:
							print(f"  → Selected camera index {idx} (likely StreamCam)")
							return idx
					else:
						cap.release()
			except Exception as e:
				# Silently continue to next backend/index
				continue
	
	# If we found working cameras but none in 1-3, return the first one
	if working_cameras:
		selected_idx, selected_backend = working_cameras[0]
		backend_name = "MSMF" if selected_backend == cv2.CAP_MSMF else ("DSHOW" if selected_backend == cv2.CAP_DSHOW else "ANY")
		print(f"  → Selected camera index {selected_idx} (backend: {backend_name})")
		return selected_idx
	
	# Fallback: return 0
	print("  ⚠ No working cameras found, defaulting to index 0")
	return 0


def open_camera(index: int = None, prefer_streamcam: bool = True) -> cv2.VideoCapture:
	"""
	Open a camera. If index is None and prefer_streamcam is True, 
	automatically detect and use Logitech StreamCam.
	"""
	import warnings
	
	# Auto-detect StreamCam if no index specified or if prefer_streamcam is True
	if (index is None or prefer_streamcam) and prefer_streamcam:
		streamcam_idx = find_logitech_streamcam()
		if streamcam_idx >= 0:
			print(f"✓ Auto-detected camera at index {streamcam_idx}")
			index = streamcam_idx
		else:
			print("⚠ Could not auto-detect StreamCam, trying default camera (index 0)")
			index = 0 if index is None else index
	
	if index is None:
		index = 0
	
	# Try backends in order: MSMF is often more reliable on Windows 10+
	backends = []
	if hasattr(cv2, "CAP_MSMF"):
		backends.append(cv2.CAP_MSMF)  # Try Media Foundation first
	if hasattr(cv2, "CAP_DSHOW"):
		backends.append(cv2.CAP_DSHOW)  # Then DirectShow
	backends.append(cv2.CAP_ANY)  # Finally, any available backend
	
	# Suppress warnings during camera opening
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=UserWarning)
		for b in backends:
			try:
				cap = cv2.VideoCapture(index, b)
				if not cap.isOpened():
					cap.release()
					continue
				ret, frm = cap.read()
				if ret and frm is not None and frm.size and float(frm.mean()) > 1.0:
					backend_name = "MSMF" if b == cv2.CAP_MSMF else ("DSHOW" if b == cv2.CAP_DSHOW else "ANY")
					print(f"✓ Opened camera {index} using {backend_name} backend")
					return cap
				cap.release()
			except Exception:
				continue
	
	raise AssertionError(f"Failed to open camera {index} with available backends")


def main():
	parser = argparse.ArgumentParser(description="ACLab: Tello Wall Corridor Follower (No Markers)")
	parser.add_argument("--camera", type=int, default=None, help="Override camera index")
	parser.add_argument("--config", type=str, default="calibration/config.yaml")
	parser.add_argument("--dry-run", action="store_true", help="Do not send Tello commands (preview mode)")
	parser.add_argument("--no-gui", action="store_true")
	parser.add_argument("--preview-only", action="store_true", help="Preview mode: show camera and checkpoints only, no drone")
	parser.add_argument("--open-route", action="store_true", help="Use open-route detector and open-path checkpoints")
	parser.add_argument("--laser-timeout", type=float, default=20.0, help="Laser follow timeout (seconds)")
	parser.add_argument("--laser-preview", action="store_true", help="Run laser detection/follow preview only (no path following)")
	args = parser.parse_args()

	cfg = load_config(args.config)
	
	# Determine camera index: command line arg > config > auto-detect StreamCam
	if args.camera is not None:
		camera_index = args.camera
		prefer_streamcam = False
		print(f"Using specified camera index: {args.camera}")
	else:
		# Auto-detect Logitech StreamCam
		print("🔍 Auto-detecting Logitech StreamCam...")
		camera_index = None  # Will be auto-detected
		prefer_streamcam = True
	
	cap = open_camera(camera_index, prefer_streamcam=prefer_streamcam)
	# Store camera settings for reconnection
	_camera_index_for_reconnect = camera_index
	_prefer_streamcam_for_reconnect = prefer_streamcam
	ret, frame = cap.read()
	assert ret and frame is not None and frame.size, "Failed to grab a frame from camera"

	# Initialize placeholder for error frames
	placeholder = np.zeros_like(frame)

	route = OpenRouteDetector(cfg.vision) if args.open_route else RouteDetector(cfg.vision)
	tracker = DroneTracker(cfg.vision, frame.shape)
	laser_detector = LaserDetector()
	# Use preview-only or dry-run mode if specified
	use_dry_run = args.dry_run or args.preview_only or args.laser_preview
	tello = TelloController(dry_run=use_dry_run)

	# If laser preview only, run detection loop and exit
	if args.laser_preview:
		print("Laser preview mode: running laser follow/detection only (no path).")
		run_laser_follow(
			cap,
			tracker,
			laser_detector,
			tello,
			cfg.control,
			timeout_s=None,  # timeout ignored; laser presence is the determinant
			show_gui=not args.no_gui,
			warmup_s=0.0,
		)
		cap.release()
		cv2.destroyAllWindows()
		return
	follower = WallFollower(cfg.control, open_route=args.open_route)
	
	# Try to connect to Tello, but don't fail if not available (for preview mode)
	if not use_dry_run:
		print("🔌 Attempting to connect to Tello drone...")
		print("   Make sure:")
		print("   1. Tello is powered on")
		print("   2. Your computer is connected to Tello Wi-Fi (TELLO-XXXXXX)")
		print("   3. Tello is ready (LED should be solid)")
		try:
			tello.connect()
			print("✓ Connected to Tello drone successfully!")
			# Verify we can actually send commands
			try:
				battery = tello.tello.get_battery() if tello.tello else None
				if battery is not None:
					print(f"✓ Tello battery: {battery}%")
			except:
				pass
		except Exception as e:
			print(f"❌ ERROR: Could not connect to Tello drone: {e}")
			print("")
			print("⚠⚠⚠ IMPORTANT: System will continue in PREVIEW MODE ⚠⚠⚠")
			print("   No commands will be sent to the drone!")
			print("")
			print("To fix the connection:")
			print("  1. Check Wi-Fi: Are you connected to Tello Wi-Fi? (TELLO-XXXXXX)")
			print("  2. Restart Tello: Turn it off and on again")
			print("  3. Check firewall: Windows Firewall may be blocking Python")
			print("  4. Close other apps: Make sure no other program is using the Tello")
			print("  5. Try again: Restart this program after fixing connection")
			print("")
			print("⚠ Continuing in preview mode (no drone commands will be sent)")
			tello = TelloController(dry_run=True)
	else:
		print("✓ Running in preview mode (no drone connection needed)")

	# Preview phase: detect path and generate checkpoints
	print("=" * 60)
	print("PREVIEW PHASE: Detecting circular track and generating path...")
	print("=" * 60)
	checkpoints = None
	preview_done = False
	detection_frame = None  # Store the frame where detection succeeded
	
	# Wait for red band detection and generate checkpoints
	while not preview_done:
		ret, frame0 = cap.read()
		if not ret or frame0 is None or not frame0.size:
			if not args.no_gui:
				frame = placeholder.copy()
				cv2.putText(frame, "No camera frame - retrying...", (20, 40),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.imshow("ACLab", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				return
			continue
		
		frame = frame0
		det = route.detect(frame)
		
		if args.open_route:
			if det is not None and det.get("centerline") is not None:
				centerline = det["centerline"].copy()
				
				if centerline.ndim == 3:
					centerline = centerline[:, 0, :]
				elif centerline.ndim == 1:
					print("⚠ Open-route: centerline wrong dimensions, skipping...")
					continue
				
				centerline = centerline.astype(np.float32)
				if centerline.shape[1] != 2 or len(centerline) < 2:
					print(f"⚠ Open-route: centerline invalid shape {centerline.shape}, skipping...")
					continue
				
				# Debug info
				print(f"  → Open centerline shape: {centerline.shape}")
				print(f"  → Centerline range: x=[{centerline[:, 0].min():.1f}, {centerline[:, 0].max():.1f}], y=[{centerline[:, 1].min():.1f}, {centerline[:, 1].max():.1f}]")
				
				try:
					checkpoints = generate_checkpoints_open(centerline, num_checkpoints=20)
					# Keep only corner/curve checkpoints: 1,4,8,11,14-20 (1-indexed)
					keep_idx = [0, 3, 7, 10, 13, 14, 15, 16, 17, 18, 19]
					if len(checkpoints) >= max(keep_idx) + 1:
						checkpoints = checkpoints[keep_idx]
					print(f"✓ Detected open track!")
					print(f"✓ Generated {len(checkpoints)} checkpoints along the open path")
					print(f"  → Checkpoints range: x=[{checkpoints[:, 0].min():.1f}, {checkpoints[:, 0].max():.1f}], y=[{checkpoints[:, 1].min():.1f}, {checkpoints[:, 1].max():.1f}]")
					# For preview, store minimal detection (centerline, band mask)
					detection_frame = {"centerline": centerline, "band_mask": det.get("band_mask")}
					preview_done = True
				except Exception as e:
					print(f"⚠ Error generating open-route checkpoints: {e}")
					continue
		else:
			if det is not None and det.get("outer") is not None:
				# Closed route: use outer ring
				outer = det["outer"].copy()
				inner_for_follower = det.get("inner") if det.get("inner") is not None else outer
				
				if outer.ndim == 3:
					outer = outer[:, 0, :]
				elif outer.ndim == 1:
					print("⚠ Warning: Outer ring has wrong dimensions, skipping...")
					continue
				
				outer = outer.astype(np.float32)
				if outer.shape[1] != 2:
					print(f"⚠ Warning: Outer ring must have 2 columns (x, y), got shape {outer.shape}")
					continue
				if len(outer) < 4:
					print(f"⚠ Warning: Outer ring has only {len(outer)} points, need at least 4")
					continue
				
				print(f"  → Outer shape: {outer.shape}")
				print(f"  → Outer range: x=[{outer[:, 0].min():.1f}, {outer[:, 0].max():.1f}], y=[{outer[:, 1].min():.1f}, {outer[:, 1].max():.1f}]")
				
				try:
					# Utilities to adjust checkpoints (straighten early legs and simplify curves)
					def _enforce_initial_block_constraints(pts: np.ndarray) -> np.ndarray:
						# Make original No.2 & No.3 same y, No.3 & No.4 same x, No.4 & No.5 same y
						q = pts.copy().astype(np.float32)
						if len(q) >= 5:
							# indices: 0-based → 1..4 => 1,2,3,4
							y23 = float((q[1, 1] + q[2, 1]) * 0.5)
							q[1, 1] = y23; q[2, 1] = y23
							x34 = float((q[2, 0] + q[3, 0]) * 0.5)
							q[2, 0] = x34; q[3, 0] = x34
							y45 = float((q[3, 1] + q[4, 1]) * 0.5)
							q[3, 1] = y45; q[4, 1] = y45
						return q

					def _angle_deg(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
						v1 = p_curr - p_prev
						v2 = p_next - p_curr
						n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
						if n1 < 1e-6 or n2 < 1e-6:
							return 0.0
						cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
						return float(np.degrees(np.arccos(cosang)))

					def _simplify_curve_points(pts: np.ndarray, angle_thresh_deg: float = 30.0, max_points: int = 4) -> np.ndarray:
						# Identify curve points where local turn angle exceeds threshold
						n = len(pts)
						if n < 6:
							return pts
						angles = []
						for i in range(n):
							p_prev = pts[(i - 1) % n]
							p_curr = pts[i]
							p_next = pts[(i + 1) % n]
							angles.append(_angle_deg(p_prev, p_curr, p_next))
						angles = np.array(angles, dtype=np.float32)
						is_curve = angles >= angle_thresh_deg
						# Find contiguous curve sequences
						keep_idx = set()
						i = 0
						while i < n:
							if is_curve[i]:
								# start of a curve run
								j = i
								run = []
								while is_curve[j % n] and len(run) < n:
									run.append(j % n)
									j += 1
									if j - i > n:
										break
								# Reduce this run to at most max_points by sampling evenly, always keep endpoints
								run_len = len(run)
								if run_len <= max_points:
									keep_idx.update(run)
								else:
									sel = np.linspace(0, run_len - 1, max_points).round().astype(int)
									for k in sel:
										keep_idx.add(run[k])
								i = j
							else:
								# straight region: keep points as-is
								keep_idx.add(i)
								i += 1
						keep_sorted = sorted(list(keep_idx))
						return pts[keep_sorted].astype(np.float32)

					checkpoints = generate_checkpoints(outer, num_checkpoints=20)
					# Apply requested adjustments: straighten early legs; simplify curves to 4 points
					checkpoints = _enforce_initial_block_constraints(checkpoints)
					checkpoints = _simplify_curve_points(
						checkpoints,
						angle_thresh_deg=float(cfg.control.curve_angle_threshold_deg),
						max_points=4
					)
					# Rotate checkpoints so starting point is lowest (max y)
					if len(checkpoints) > 0:
						start_idx = int(np.argmax(checkpoints[:, 1]))
						checkpoints = np.vstack([checkpoints[start_idx:], checkpoints[:start_idx]])
					print(f"✓ Detected circular track (outer ring)!")
					print(f"✓ Generated {len(checkpoints)} checkpoints along the outer path")
					print(f"  → Checkpoints range: x=[{checkpoints[:, 0].min():.1f}, {checkpoints[:, 0].max():.1f}], y=[{checkpoints[:, 1].min():.1f}, {checkpoints[:, 1].max():.1f}]")
					detection_frame = det.copy()
					detection_frame["inner"] = inner_for_follower
					preview_done = True
				except Exception as e:
					print(f"⚠ Error generating checkpoints: {e}")
					print(f"  → Outer shape: {outer.shape}, dtype: {outer.dtype}")
					continue
			
			# PREVIOUS (centerline-based) logic kept for reference:
			# if det["centerline"] is not None:
			# 	centerline = det["centerline"].copy()
			# 	if centerline.ndim == 3:
			# 		centerline = centerline[:, 0, :]
			# 	elif centerline.ndim == 1:
			# 		continue
			# 	centerline = centerline.astype(np.float32)
			# 	if centerline.shape[1] != 2 or len(centerline) < 4:
			# 		continue
			# 	checkpoints = generate_checkpoints(centerline, num_checkpoints=20)
		
		if det is None:
			# Show preview with detection status
			if not args.no_gui:
				status = "Detecting red path (open)..." if args.open_route else "Detecting red circle..."
				cv2.putText(frame, status, (20, 40),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
				cv2.putText(frame, "Press 'q' to quit", (20, 80),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
				cv2.imshow("ACLab", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				return
			continue
	
	# Show checkpoints and wait for 'r' to run
	print("=" * 60)
	print("PATH PREVIEW: Showing projected path with checkpoints")
	print("Press 'r' to start following the path")
	print("Press 'q' to quit")
	print("=" * 60)
	
	run_started = False
	while not run_started:
		ret, frame0 = cap.read()
		if not ret or frame0 is None or not frame0.size:
			if not args.no_gui:
				frame = placeholder.copy()
				cv2.putText(frame, "No camera frame - retrying...", (20, 40),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.imshow("ACLab", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				return
			continue
		
		frame = frame0
		# Use the stored detection for display (the one that generated checkpoints)
		# This ensures checkpoints align with displayed centerline
		det = detection_frame if detection_frame is not None else route.detect(frame)
		
		if not args.no_gui:
			# Get blue mask for preview too
			blue_mask_preview = None
			if det is not None:
				blue_mask_preview = tracker.get_debug_mask(frame, det["band_mask"])
			# Draw overlays with checkpoints
			draw_overlays(frame, det, None, checkpoints=checkpoints, blue_mask=blue_mask_preview, open_route=args.open_route)
			label = "PREVIEW: Open Path (20 checkpoints)" if args.open_route else "PREVIEW: Projected Path (outer ring, 20 checkpoints)"
			cv2.putText(frame, label, (10, 24),
			           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
			cv2.putText(frame, "Press 'r' to RUN | Press 'q' to quit", (10, 50),
			           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
			cv2.imshow("ACLab", frame)
		
		key = cv2.waitKey(1) & 0xFF
		if key == ord('r'):
			run_started = True
			print("Starting drone flight...")
		elif key == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			return
	
	# Set checkpoints in follower
	# Start from checkpoint No.2 and never visit original No.1 (drop it)
	if checkpoints is not None and len(checkpoints) > 1:
		checkpoints = checkpoints[1:].copy()
	follower.set_checkpoints(checkpoints)
	
	print("Press 't' in the video window to take off. No other keys are active during flight.")
	print("Press 'q' in the video window for emergency stop and land.")
	last = time.time()
	armed = False
	hover_until = None  # time until which we should hover after takeoff
	last_log = 0.0
	frame_fail_count = 0
	def _zero_cmd():
		return type("Cmd", (), {"lr":0,"fb":0,"ud":0,"yaw":0})()
	def _nudge_forward(tello_ctrl: TelloController, fb:int=15, duration_s:float=0.6):
		"""
		Give a gentle forward command after takeoff to cancel hardware bias.
		Sends small forward RC for a brief duration, then neutral.
		"""
		fb = max(-30, min(30, int(fb)))
		start = time.time()
		while time.time() - start < duration_s:
			tello_ctrl.send_rc(0, fb, 0, 0)
			time.sleep(0.06)
		tello_ctrl.send_rc(0, 0, 0, 0)
	laser_phase_started = False
	while True:
		ret, frame0 = cap.read()
		if not ret or frame0 is None or not frame0.size:
			frame_fail_count += 1
			# Try to reopen camera after brief failures
			if frame_fail_count % (cfg.control.frame_rate_hz * 2) == 0:
				try:
					cap.release()
				except Exception:
					pass
				try:
					# Reconnect using the same camera settings
					cap = open_camera(_camera_index_for_reconnect, prefer_streamcam=_prefer_streamcam_for_reconnect)
				except Exception:
					pass
			# Keep window alive with placeholder
			frame = placeholder.copy()
			cv2.putText(frame, "No camera frame - retrying...", (20, 40),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
			det = None
			drone_xy = None
			blue_mask_debug = None
		else:
			frame_fail_count = 0
			frame = frame0
			det = route.detect(frame)
			drone_xy = None
			blue_mask_debug = None
			if det is not None:
				drone_xy = tracker.detect(frame, det["band_mask"])
				# Get blue mask for visualization
				blue_mask_debug = tracker.get_debug_mask(frame, det["band_mask"])
			else:
				# Even if red band not detected, try to detect blue (for debugging)
				blue_mask_debug = tracker.get_debug_mask(frame, np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255)

		# Controller runs only after takeoff
		if armed:
			now_t = time.time()
			if hover_until is not None and now_t < hover_until:
				# Hold hover to allow vision to lock on drone
				state, cmd = ("HOVER_WAIT", _zero_cmd())
			elif laser_phase_started:
				# After laser follow, just hold hover until external stop
				state, cmd = ("LASER_DONE", _zero_cmd())
			else:
				hover_until = None
				if args.open_route:
					# Open route: rely on checkpoints and ignore outer/inner
					centerline_for_step = None
					outer_for_step = None
					inner_for_step = None
				else:
					# Closed route: outer/inner available for entry
					centerline_for_step = None if checkpoints is not None else (None if det is None else det.get("centerline"))
					outer_for_step = None if det is None else det.get("outer")
					inner_for_step = None if det is None else det.get("inner")
				state, cmd = follower.step(
					centerline_for_step,
					outer_for_step,
					inner_for_step,
					drone_xy,
				)
		else:
			state, cmd = (State.TAKEOFF, _zero_cmd())

		# Periodic console log for visibility
		if time.time() - last_log > 1.0:
			drone_pos_str = f"({drone_xy[0]:.0f},{drone_xy[1]:.0f})" if drone_xy is not None else "None"
			print(f"[state={state}] det={'Y' if det is not None else 'N'} "
			      f"drone={drone_pos_str} "
			      f"cmd=({cmd.lr},{cmd.fb},{cmd.ud},{cmd.yaw}) armed={armed}")
			last_log = time.time()

		# Map command to Tello only when airborne (armed)
		if armed:
			if state in (State.DONE, State.ABORT):
				tello.send_rc(0, 0, 0, 0)
			else:
				tello.send_rc(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

		if not args.no_gui:
			current_checkpoint_idx = follower.get_current_checkpoint_idx() if hasattr(follower, 'get_current_checkpoint_idx') else None
			draw_overlays(frame, det, drone_xy, checkpoints=checkpoints, current_checkpoint_idx=current_checkpoint_idx, blue_mask=blue_mask_debug, open_route=args.open_route)
			cv2.putText(frame, f"State: {state}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
			if drone_xy is None:
				cv2.putText(frame, "Drone: NOT DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			else:
				cv2.putText(frame, f"Drone: ({drone_xy[0]}, {drone_xy[1]})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			cv2.imshow("ACLab", frame)
			key = cv2.waitKey(1) & 0xFF
			# Emergency stop: 'q' key lands immediately
			if key == ord('q'):
				print("\n⚠ EMERGENCY STOP: Landing drone immediately...")
				if armed:
					tello.send_rc(0, 0, 0, 0)  # Stop all movement
					tello.land()  # Land immediately
					print("✓ Land command sent")
				cap.release()
				cv2.destroyAllWindows()
				return
			# Only allow 't' to arm/takeoff when not already airborne
			if not armed and key == ord('t'):
				if tello.dry:
					print("❌ ERROR: Cannot takeoff - system is in dry-run/preview mode!")
					print("   The Tello connection failed earlier. Please:")
					print("   1. Check Tello Wi-Fi connection")
					print("   2. Restart the Tello")
					print("   3. Run the program again without --dry-run flag")
				else:
					print("🚁 Taking off...")
					try:
						tello.takeoff()
						armed = True
						hover_until = time.time() + cfg.control.takeoff_hover_s
						# Gentle forward nudge to cancel backward drift
						_nudge_forward(tello, fb=12, duration_s=0.6)
						print(f"🛑 Hovering for {cfg.control.takeoff_hover_s:.1f}s to let vision detect the drone...")
						print("✓ Takeoff command sent!")
					except Exception as e:
						print(f"❌ Takeoff failed: {e}")
						print("   The drone may not be ready. Check battery and connection.")
		# Auto actions
		if state == State.DONE:
			if args.open_route and not laser_phase_started:
				# Start laser follow phase instead of immediate landing
				print("Open-route completed. Starting laser follow phase...")
				laser_reason = run_laser_follow(
					cap,
					tracker,
					laser_detector,
					tello,
					cfg.control,
					timeout_s=None,  # timeout ignored; laser presence is the determinant
					show_gui=not args.no_gui,
					warmup_s=2.0,
				)
				laser_phase_started = True
				if laser_reason == "no_laser":
					# Already landed in laser follow; exit loop
					armed = False
					break
				# After laser follow, hold position; do not land
				state = "LASER_DONE"
				cmd = _zero_cmd()
				continue
			tello.land()
			armed = False
			break
		if state == State.ABORT:
			tello.land()
			armed = False
			break

		# Maintain loop rate approximately
		now = time.time()
		sleep_time = max(0.0, (1.0 / cfg.control.frame_rate_hz) - (now - last))
		if sleep_time > 0:
			time.sleep(sleep_time)
		last = time.time()
	cap.release()
	cv2.destroyAllWindows()
	# Final safety check - land if still armed
	if 'armed' in locals() and armed:
		print("⚠ Final safety check: Ensuring drone lands...")
		tello.send_rc(0, 0, 0, 0)
		time.sleep(0.1)
		tello.land()
		time.sleep(1)


if __name__ == "__main__":
	main()


