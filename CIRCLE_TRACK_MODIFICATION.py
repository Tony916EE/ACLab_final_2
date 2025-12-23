"""
Example modification for circular track support.

This file shows how to modify the _compute_entry() method in wall_follow.py
to work with circular tracks instead of square tracks.
"""

import cv2
import numpy as np
from utils.geometry import point_in_ring


def _compute_entry_circle(self, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """
    Modified entry point calculation for circular tracks.
    
    Instead of using "bottom-right corner" (which doesn't exist for circles),
    this version uses the bottom-most point and moves inward toward the center.
    
    This works for BOTH circles and squares, making it more general.
    """
    # Handle different array shapes
    if outer.ndim == 3:
        outer = outer[:, 0, :]
    if inner.ndim == 3:
        inner = inner[:, 0, :]
    
    # Find bottom-most point (maximum y coordinate)
    # This works for circles (bottom of circle) and squares (bottom-right corner)
    bottom_idx = np.argmax(outer[:, 1])
    br = outer[bottom_idx]
    
    # Calculate center of the ring
    center = np.mean(outer, axis=0)
    
    # Direction from bottom point toward center
    direction = center - br
    direction_norm = np.linalg.norm(direction)
    
    # Normalize direction vector (avoid division by zero)
    if direction_norm > 1e-6:
        direction = direction / direction_norm
    else:
        # Fallback: use downward direction if center calculation fails
        direction = np.array([0.0, -1.0], dtype=np.float32)
    
    # Calculate offset distance (same as original)
    bb = cv2.boundingRect(outer.astype(np.int32))
    shorter = min(bb[2], bb[3])
    offset = self.cfg.homing_offset_frac * shorter
    
    # Move inward from bottom point toward center
    entry = br + direction * offset
    
    # Ensure entry is inside the ring (same as original)
    for _ in range(10):
        if point_in_ring(outer, inner, (float(entry[0]), float(entry[1]))):
            break
        entry = (entry + center) / 2.0
    
    return entry


def _compute_entry_circle_alternative(self, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """
    Alternative: Use right-most point instead of bottom-most.
    
    Choose this if you want the drone to start from the right side of the circle.
    """
    if outer.ndim == 3:
        outer = outer[:, 0, :]
    
    # Find right-most point (maximum x coordinate)
    right_idx = np.argmax(outer[:, 0])
    br = outer[right_idx]
    
    # Move inward toward center
    center = np.mean(outer, axis=0)
    direction = center - br
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm > 1e-6:
        direction = direction / direction_norm
    else:
        direction = np.array([-1.0, 0.0], dtype=np.float32)
    
    bb = cv2.boundingRect(outer.astype(np.int32))
    shorter = min(bb[2], bb[3])
    offset = self.cfg.homing_offset_frac * shorter
    
    entry = br + direction * offset
    
    # Ensure entry is inside the ring
    for _ in range(10):
        if point_in_ring(outer, inner, (float(entry[0]), float(entry[1]))):
            break
        entry = (entry + center) / 2.0
    
    return entry


# ============================================================================
# HOW TO APPLY THIS MODIFICATION
# ============================================================================
#
# 1. Open: ACLab/tony_code/control/wall_follow.py
#
# 2. Find the _compute_entry() method (around line 64)
#
# 3. Replace the method with _compute_entry_circle() above
#    (remove the 'self' parameter from the function definition)
#
# 4. The modified method should look like:
#
#    def _compute_entry(self, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
#        if outer.ndim == 3:
#            outer = outer[:, 0, :]
#        
#        # Find bottom-most point
#        bottom_idx = np.argmax(outer[:, 1])
#        br = outer[bottom_idx]
#        
#        # Move inward toward center
#        center = np.mean(outer, axis=0)
#        direction = center - br
#        direction_norm = np.linalg.norm(direction)
#        
#        if direction_norm > 1e-6:
#            direction = direction / direction_norm
#        else:
#            direction = np.array([0.0, -1.0], dtype=np.float32)
#        
#        bb = cv2.boundingRect(outer.astype(np.int32))
#        shorter = min(bb[2], bb[3])
#        offset = self.cfg.homing_offset_frac * shorter
#        
#        entry = br + direction * offset
#        
#        # Ensure entry is inside the ring
#        for _ in range(10):
#            if point_in_ring(outer, inner, (float(entry[0]), float(entry[1]))):
#                break
#            entry = (entry + center) / 2.0
#        
#        return entry
#
# 5. Test with: python -m scripts.offline_test --camera 0
#
# ============================================================================
