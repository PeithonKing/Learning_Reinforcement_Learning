from svgpathtools import svg2paths
import numpy as np
import matplotlib.pyplot as plt

# Canvas settings
WIDTH = 800
HEIGHT = 500
PADDING = 75  # on all 4 sides

# If True, scale uniformly (preserve aspect ratio) using the smaller available dimension.
# If False, keep existing behavior (independent scaling in X and Y).
keep_aspect_ratio = True

filename = "rounded_square"
# filename = "hexagon"

# Load SVG paths
paths, attributes = svg2paths(f"{filename}.svg")

# Sample waypoints
waypoints = []
for path in paths:
    for t in np.linspace(0, 1, 1000):
        point = path.point(t)
        waypoints.append((point.real, point.imag))

waypoints = np.array(waypoints)
# np.savetxt(f"{filename}_waypoints.txt", waypoints)
np.save(f"{filename}_waypoints.npy", waypoints)

# --- Normalization (scale to 0..1) ---
min_x, min_y = waypoints[:,0].min(), waypoints[:,1].min()
max_x, max_y = waypoints[:,0].max(), waypoints[:,1].max()

if not keep_aspect_ratio:
    # Existing behavior: normalize each axis to [0,1] independently and scale to canvas
    waypoints[:,0] = (waypoints[:,0] - min_x) / (max_x - min_x)
    waypoints[:,1] = (waypoints[:,1] - min_y) / (max_y - min_y)

    # --- Scale to canvas (with padding) ---
    waypoints[:,0] = PADDING + waypoints[:,0] * (WIDTH - 2*PADDING)
    waypoints[:,1] = PADDING + waypoints[:,1] * (HEIGHT - 2*PADDING)
else:
    # Preserve aspect ratio: use a uniform scale so the larger original dimension fits within the available area
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    avail_w = WIDTH - 2 * PADDING
    avail_h = HEIGHT - 2 * PADDING

    # Handle degenerate cases where width or height could be zero
    scale_w = avail_w / bbox_w if bbox_w != 0 else float('inf')
    scale_h = avail_h / bbox_h if bbox_h != 0 else float('inf')
    scale = min(scale_w, scale_h) if (scale_w != float('inf') or scale_h != float('inf')) else 1.0

    # Translate to origin then scale uniformly
    waypoints[:,0] = (waypoints[:,0] - min_x) * scale
    waypoints[:,1] = (waypoints[:,1] - min_y) * scale

    # Center within the padded area
    used_w = bbox_w * scale
    used_h = bbox_h * scale
    offset_x = PADDING + (avail_w - used_w) / 2.0
    offset_y = PADDING + (avail_h - used_h) / 2.0
    waypoints[:,0] += offset_x
    waypoints[:,1] += offset_y

np.save(f"../tracks/{filename}_waypoints.npy", waypoints)
# --- Flip Y-axis (to match screen coordinates like pygame) ---
waypoints[:,1] = HEIGHT - waypoints[:,1]


# --- Plot result ---
plt.figure(figsize=(WIDTH/100, HEIGHT/100))
plt.plot(waypoints[:, 0], waypoints[:, 1], "ko", markersize=20)
plt.axis("equal")

plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)

# Remove axes, ticks, background
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding around plot

plt.savefig(f"../tracks/{filename}.png", dpi=100)
