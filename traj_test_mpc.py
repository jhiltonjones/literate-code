import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def angle_wrap_deg(a):
    """Wrap to [-180, 180)."""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def unwrap_deg(seq):
    return np.degrees(np.unwrap(np.radians(seq)))

def click_points(image_rgb, title, n=None):
    plt.figure(figsize=(9, 7))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    pts = plt.ginput(n=n if n is not None else -1, timeout=0)
    plt.close()
    if n is not None and len(pts) != n:
        raise ValueError(f"Expected {n} points, got {len(pts)}.")
    if len(pts) < 2:
        raise ValueError("Need at least two points.")
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    return xs, ys

def compute_centerline_angle_deg(p1, p2):
    # image coords: x right, y down. For angles, we can stick to this convention consistently.
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))  # 0° is right, +CCW is "downwards" in image coords

def tangent_angles_for_polyline(xs, ys):
    """Per-point tangent angle (deg) for a clicked polyline (image coords)."""
    # finite differences in image coordinates
    dx = np.zeros_like(xs)
    dy = np.zeros_like(ys)

    if len(xs) == 2:
        dx[:] = xs[1] - xs[0]
        dy[:] = ys[1] - ys[0]
    else:
        dx[0]  = xs[1]   - xs[0]
        dy[0]  = ys[1]   - ys[0]
        dx[-1] = xs[-1]  - xs[-2]
        dy[-1] = ys[-1]  - ys[-2]
        dx[1:-1] = xs[2:] - xs[:-2]
        dy[1:-1] = ys[2:] - ys[:-2]

    ang = np.degrees(np.arctan2(dy, dx))
    return ang

def save_csv(path, xs, ys, ang_abs, ang_from_center):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("idx,x_px,y_px,tangent_deg,angle_from_center_deg\n")
        for i,(x,y,a,b) in enumerate(zip(xs,ys,ang_abs,ang_from_center)):
            f.write(f"{i},{x:.3f},{y:.3f},{a:.6f},{b:.6f}\n")
    print(f"[OK] saved {path}")

if __name__ == "__main__":
    IMG_PATH = "focused_image.jpg"  # change if needed
    bgr = cv2.imread(IMG_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {IMG_PATH}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1) Click two points for the centerline
    cx, cy = click_points(rgb, "Click TWO points to define the centerline", n=2)
    center_ang_deg = compute_centerline_angle_deg((cx[0], cy[0]), (cx[1], cy[1]))
    print(f"[INFO] Centerline angle: {center_ang_deg:.3f} deg (image coords)")

    # 2) Click points along the curve (ENTER to finish)
    xs, ys = click_points(rgb, "Click curve points in order (ENTER to finish)")

    # 3) Tangent angles at each clicked point (absolute, in image coords)
    tan_abs_deg = tangent_angles_for_polyline(xs, ys)

    # 4) Angle from the centerline (unwrap to keep continuity)
    delta_deg = angle_wrap_deg(tan_abs_deg - center_ang_deg)
    delta_deg_unwrapped = unwrap_deg(delta_deg)

    # 5) Save results
    OUT = "outputs/angles_from_centerline.csv"
    # save_csv(OUT, xs, ys, tan_abs_deg, delta_deg_unwrapped)

    # 6) Visualize
    plt.figure(figsize=(9,7))
    plt.imshow(rgb)
    # plot clicked curve
    plt.plot(xs, ys, 'o-', ms=4, lw=1.5, label="curve points")

    # draw centerline
    plt.plot(cx, cy, 's--', lw=2, label=f"centerline ({center_ang_deg:.1f}°)")

    # draw small tangent ticks at each clicked point
    L = 25  # pixels (visual length)
    th = np.radians(tan_abs_deg)
    dx = L*np.cos(th)
    dy = L*np.sin(th)  # image coords: y down -> same sign as angle definition above
    for xi, yi, dxi, dyi in zip(xs, ys, dx, dy):
        plt.plot([xi, xi+dxi], [yi, yi+dyi], '-', lw=1)

    plt.legend(loc="lower right")
    plt.title("Angles from centerline at clicked points")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # quick print
    print("\nIndex | tangent_deg | angle_from_center_deg (unwrapped)")
    for i in range(len(xs)):
        print(f"{i:5d} | {tan_abs_deg[i]:12.3f} | {delta_deg_unwrapped[i]:26.3f}")
