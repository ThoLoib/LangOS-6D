import sys
import numpy as np
import plotly.graph_objects as go

if len(sys.argv) < 2:
    print("usage: pc_viewer.py <path-to.npz> [out.html]")
    sys.exit(1)

npz_path = sys.argv[1]
out_html = sys.argv[2] if len(sys.argv) > 2 else "object_images/pc_viewer.html"

d = np.load(npz_path)
key_x = "points" if "points" in d else list(d.keys())[0]
pts = d[key_x]
cols = d["colors"] if "colors" in d else None

if cols is not None:
    c = cols if cols.max() <= 1.0001 else cols / 255.0
    c = np.clip(c, 0, 1)
    color_strs = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in c]
else:
    color_strs = pts[:, 2]

fig = go.Figure(data=[go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode="markers",
    marker=dict(size=2, color=color_strs),
)])
fig.update_layout(
    scene=dict(aspectmode="data"),
    margin=dict(l=0, r=0, b=0, t=30),
    title=npz_path,
)
fig.write_html(out_html)
print("saved", out_html)
