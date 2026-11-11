"""Computer vision on rendered overhead-camera images (MuJoCo offscreen renderer, EGL).
HSV thresholding finds each flask cap; pixel + rendered depth -> world position;
a "cap present" check gates every grasp."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from sim import Workcell

# HSV ranges (OpenCV: H 0-179). Red wraps around 0.
CAP_HSV = {
    "red": [((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (179, 255, 255))],
    "blue": [((105, 120, 60), (130, 255, 255))],
    "yellow": [((20, 120, 80), (35, 255, 255))],
}
MIN_CAP_AREA_PX = 40  # a visible cap is ~90 px at 640x480 from this camera; below 40 = absent


@dataclass
class CapDetection:
    present: bool
    area_px: int
    pixel: tuple[float, float] | None = None
    world: np.ndarray | None = None
    reason: str = ""


class OverheadCamera:
    def __init__(self, wc: Workcell, name="overhead", width=640, height=480):
        self.wc = wc
        self.cam_id = wc.m.camera(name).id
        self.name = name
        self.w, self.h = width, height
        self.rgb_r = mujoco.Renderer(wc.m, height, width)
        self.depth_r = mujoco.Renderer(wc.m, height, width)
        self.depth_r.enable_depth_rendering()
        fovy = np.deg2rad(wc.m.cam_fovy[self.cam_id])
        self.f = (height / 2) / np.tan(fovy / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2

    def capture(self):
        mujoco.mj_forward(self.wc.m, self.wc.d)
        self.rgb_r.update_scene(self.wc.d, self.name)
        rgb = self.rgb_r.render().copy()
        self.depth_r.update_scene(self.wc.d, self.name)
        depth = self.depth_r.render().copy()
        return rgb, depth

    def pixel_to_world(self, u, v, depth_m):
        d = self.wc.d
        pos = d.cam_xpos[self.cam_id]
        rot = d.cam_xmat[self.cam_id].reshape(3, 3)
        # MuJoCo camera frame: x right, y up, looks along -z
        p_cam = np.array([(u - self.cx) / self.f * depth_m, -(v - self.cy) / self.f * depth_m, -depth_m])
        return pos + rot @ p_cam

    def world_to_pixel(self, p):
        d = self.wc.d
        pos = d.cam_xpos[self.cam_id]
        rot = d.cam_xmat[self.cam_id].reshape(3, 3)
        pc = rot.T @ (np.asarray(p) - pos)
        z = -pc[2]
        return self.cx + self.f * pc[0] / z, self.cy - self.f * pc[1] / z

    def close(self):
        self.rgb_r.close()
        self.depth_r.close()


def cap_mask(rgb, color):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in CAP_HSV[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_cap(cam: OverheadCamera, rgb, depth, color, expected_world, roi_px=60) -> CapDetection:
    """Look for a cap of `color` near the expected slot position (region of interest).
    Returns the cap top-centre in world coordinates from the depth image."""
    mask = cap_mask(rgb, color)
    eu, ev = cam.world_to_pixel(expected_world)
    roi = np.zeros_like(mask)
    cv2.circle(roi, (int(round(eu)), int(round(ev))), roi_px, 255, -1)
    mask &= roi
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return CapDetection(False, 0, reason="no_cap_pixels")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < MIN_CAP_AREA_PX:
        return CapDetection(False, area, reason="cap_blob_too_small")
    ys, xs = np.nonzero(labels == k)
    u, v = float(cents[k][0]), float(cents[k][1])
    z = float(np.median(depth[ys, xs]))
    return CapDetection(True, area, (u, v), cam.pixel_to_world(u, v, z))
