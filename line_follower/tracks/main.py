from svgpathtools import svg2paths
import numpy as np
import matplotlib.pyplot as plt

# Canvas settings
WIDTH = 800
HEIGHT = 500
PADDING = 75  # on all 4 sides

filename = "path"

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

waypoints[:,0] = (waypoints[:,0] - min_x) / (max_x - min_x)
waypoints[:,1] = (waypoints[:,1] - min_y) / (max_y - min_y)

# --- Scale to canvas (with padding) ---
waypoints[:,0] = PADDING + waypoints[:,0] * (WIDTH - 2*PADDING)
waypoints[:,1] = PADDING + waypoints[:,1] * (HEIGHT - 2*PADDING)

np.save(f"../tracks/{filename}_waypoints.npy", waypoints)
# --- Flip Y-axis (to match screen coordinates like pygame) ---
waypoints[:,1] = HEIGHT - waypoints[:,1]


# --- Plot result ---
plt.figure(figsize=(WIDTH/100, HEIGHT/100))
plt.plot(waypoints[:, 0], waypoints[:, 1], "k", linewidth=20)
plt.axis("equal")

# # Draw canvas border for reference
# plt.axvline(x=0, color='k', linestyle='--')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.axvline(x=WIDTH, color='k', linestyle='--')
# plt.axhline(y=HEIGHT, color='k', linestyle='--')
# plt.axvline(x=PADDING, color='r', linestyle='--')
# plt.axhline(y=PADDING, color='r', linestyle='--')
# plt.axvline(x=WIDTH-PADDING, color='r', linestyle='--')
# plt.axhline(y=HEIGHT-PADDING, color='r', linestyle='--')

plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)

# Remove axes, ticks, background
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding around plot

plt.savefig(f"../tracks/{filename}.png", dpi=100)
