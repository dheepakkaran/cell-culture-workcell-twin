"""Save two pictures of the workcell: the overhead camera image the vision step uses
(with detected caps marked) and a view into the hood with the arm carrying a flask."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from behaviors import load_protocol  # noqa: E402
from planner import ik_dls  # noqa: E402
from sim import FINGER_CLOSED, ROOT, Workcell  # noqa: E402
from vision import OverheadCamera, detect_cap  # noqa: E402

out = ROOT / "results"
wc = Workcell()
p = load_protocol()
cam = OverheadCamera(wc)
wc.remove_cap("flask_C")  # show the rejection case
rgb, depth = cam.capture()
img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
for lw in p["labware"]:
    exp = np.array(p["slots"][lw["source_slot"]]) + [0, 0, 0.095]
    det = detect_cap(cam, rgb, depth, lw["cap_color"], exp)
    u, v = cam.world_to_pixel(exp)
    col = (0, 200, 0) if det.present else (0, 0, 255)
    cv2.circle(img, (int(u), int(v)), 18, col, 2)
    cv2.putText(img, f"{lw['id']}: {'cap OK' if det.present else 'NO CAP -> reject'}",
                (int(u) - 70, int(v) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
cv2.imwrite(str(out / "overhead_vision.png"), img)

# arm carrying flask_A over the divider
wc.reset()
wc.ignore_flask = "flask_A"
q, _ = ik_dls(wc, [0.36, 0.32, 0.165])
wc.ignore_flask = None
wc.fingers = FINGER_CLOSED
wc.set_q(q)
wc.attach("flask_A")
q2, _ = ik_dls(wc, [0.42, 0.10, 0.62], q0=q)
wc.set_q(q2 if q2 is not None else q)
mujoco.mj_forward(wc.m, wc.d)
r = mujoco.Renderer(wc.m, 480, 640)
fc = mujoco.MjvCamera()
fc.lookat[:] = [0.5, 0, 0.2]
fc.distance, fc.azimuth, fc.elevation = 0.9, 0, -62
r.update_scene(wc.d, fc)
cv2.imwrite(str(out / "hood_view.png"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
print("wrote", out / "overhead_vision.png", out / "hood_view.png")
