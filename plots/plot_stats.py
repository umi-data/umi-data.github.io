import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from interactive_annotate import apply_config

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Data (from index.html) ────────────────────────────────────────────────────
projects = [
    ("UMI",                "2024-02-15", 258 + 249 + 284 + 305 + 1447),
    ("ManiWav",            "2024-06-27", 119 + 283 + 145 + 193 + 274),
    ("UMI on Legs",        "2024-07-14", 14 + 500),
    ("Fast-UMI",           "2024-09-29", 251+243+546+499+500+512+100+429
                                        +398+517+100+500+496+833+499+895
                                        +608+20+20+495+300+204),
    ("Data Scaling Laws",  "2024-10-24", 3648+6896+3564+6505+1752+1733),
    ("LEGATO",             "2024-11-06", 150*6),
    ("ViTaMIn",            "2025-04-08", 73+138+125+137+83+101+83+101),
    ("DexWild",            "2025-05-12", 1030+248+1123+295+2820+388+2323+546+621+111),
    ("DexUMI",             "2025-05-28", 309+175+404+440+464),
    ("Touch in the Wild",  "2025-07-20", 38+14+164+51+24+26+40+275+50+244+192+266
                                        +150+188+107+202+114+239+150+203),
    ("exUMI",              "2025-09-18", 208+170+135+201+139+270+256+281),
    ("MV-UMI",             "2025-09-23", 454+454+199+263),
    ("ManipForce",         "2025-09-23", 101+110+69+102+108+107),
    ("FastUMI-100K",       "2025-10-09", 3368+3343+3611+3156+4036+2541+3104+2869
                                        +4206+4384+4137+3244+3748+3952+2147+2864
                                        +3004+2926+3136+2569+1809+2539+3323+2035
                                        +4244+1265+2974+1740+2161+2161+2227),
    ("ViTaMIn-B",          "2025-11-08", 123+345+195+181),
    ("HuMI",               "2026-02-06", 103+105+105+104+410),
    ("TAMEn",              "2026-04-24", 271+159+148+146),
    ("UMI-3D",             "2026-04-24", 3500+769+340),
]

projects   = sorted(projects, key=lambda x: x[1])
names      = [p[0] for p in projects]
dates      = [datetime.strptime(p[1], "%Y-%m-%d") for p in projects]
demos      = [p[2] for p in projects]
cumulative = np.cumsum(demos)

t0         = datetime(2024, 1, 15)
plot_dates = [t0] + dates
plot_cum   = [0.0] + [v / 1000 for v in cumulative]

# ── Style ─────────────────────────────────────────────────────────────────────
PRIMARY = "#5469d4"
LIGHT   = "#b8c2f0"
MONO    = "monospace"

# ── Figure & axes ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
ax.set_facecolor("white")

ax.plot(plot_dates, plot_cum,
        color=PRIMARY, linewidth=3, zorder=2, solid_capstyle="round")

cx = dates
cy = [v / 1000 for v in cumulative]
ax.scatter(cx, cy, color=PRIMARY, s=80, edgecolors=LIGHT, linewidths=3, zorder=3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)

ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.xticks(fontfamily=MONO, fontsize=18, fontweight="bold")
plt.yticks(fontfamily=MONO, fontsize=18)

ax.set_ylabel("Thousands of Demos", fontsize=22, fontweight="bold", fontfamily=MONO)
ax.set_title("Cumulative UMI Demos over Time     ",
             fontsize=28, fontweight="bold", fontfamily=MONO, pad=20)

x_max = dates[-1] + timedelta(days=120)
ax.set_xlim(t0, x_max)
ax.set_ylim(0, max(cumulative) / 1000 * 1.05)

# ═══════════════════════════════════════════════════════════════════════════════
#  Smart annotation layout
# ═══════════════════════════════════════════════════════════════════════════════

def seg_intersect(a, b, c, d):
    """True if open segment a-b and open segment c-d intersect."""
    def cross2(u, v): return u[0]*v[1] - u[1]*v[0]
    r, s = b - a, d - c
    denom = cross2(r, s)
    if abs(denom) < 1e-9:
        return False
    t = cross2(c - a, s) / denom
    u = cross2(c - a, r) / denom
    return 1e-6 < t < 1-1e-6 and 1e-6 < u < 1-1e-6


def auto_layout(ax, fig, dates, cy, t0,
                base_dist_px=110,
                min_sep_px=38,
                max_iter=800,
                repulsion_k=1.5):
    """
    Compute label positions that are:
      • along the outward normal at each turning point
      • well-separated (min_sep_px apart in display space)
      • non-crossing leader lines (fixed by mirroring)
      • kept within the axes bounding box

    Returns list of (x_label_datetime, y_label_data, ha_str) per point.
    """
    fig.canvas.draw()
    T    = ax.transData
    Tinv = T.inverted()
    n    = len(dates)

    # Data-point positions in display (pixel) space
    pts = np.array([T.transform((mdates.date2num(d), y))
                    for d, y in zip(dates, cy)])

    # Full polyline including the origin
    origin = T.transform((mdates.date2num(t0), 0.0))
    path   = np.vstack([origin, pts])           # (n+1, 2)

    # Axes bounding box in display coords (for boundary clamping)
    bbox = ax.get_window_extent()
    x_lo, x_hi = bbox.x0, bbox.x1
    y_lo, y_hi = bbox.y0, bbox.y1

    # ── Step 1: outward normal at every data point ────────────────────────────
    normals = np.zeros((n, 2))
    for i in range(n):
        j      = i + 1                          # index into path[]
        v_in   = path[j] - path[j - 1]
        v_in  /= np.linalg.norm(v_in) + 1e-9
        v_out  = (path[j + 1] - path[j]) if j + 1 < len(path) else v_in
        v_out /= np.linalg.norm(v_out) + 1e-9

        tangent = v_in + v_out
        tlen    = np.linalg.norm(tangent)
        tangent = tangent / tlen if tlen > 1e-9 else v_in

        n_left  = np.array([-tangent[1],  tangent[0]])   # CCW 90°
        n_right = np.array([ tangent[1], -tangent[0]])   # CW  90°

        # Cross product of incoming × outgoing:
        # > 0 (CCW / left turn) → convex side is RIGHT
        # < 0 (CW  / right turn)→ convex side is LEFT
        cross = v_in[0]*v_out[1] - v_in[1]*v_out[0]

        # Pick the normal that points more "upward" when the cross is near zero
        # (nearly collinear segments → prefer upward-left)
        if abs(cross) < 0.05:
            # Use the normal that has a larger upward component
            candidate = n_left if n_left[1] > n_right[1] else n_right
        else:
            candidate = n_right if cross > 0 else n_left

        normals[i] = candidate

    # ── Step 2: initial label positions along the normal ──────────────────────
    lpos = pts + normals * base_dist_px

    # ── Step 3: clamp labels into the axes bbox (with margin) ─────────────────
    MARGIN = 8   # px
    def clamp(lp):
        lp[:, 0] = np.clip(lp[:, 0], x_lo + MARGIN, x_hi - MARGIN)
        lp[:, 1] = np.clip(lp[:, 1], y_lo + MARGIN, y_hi - MARGIN)

    clamp(lpos)

    # ── Step 4: repulsion loop ─────────────────────────────────────────────────
    def repulsion_pass(lp, iterations):
        for _ in range(iterations):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    diff = lp[i] - lp[j]
                    dist = np.linalg.norm(diff)
                    if dist < min_sep_px:
                        push      = (min_sep_px - dist) * repulsion_k / 2 + 0.5
                        direction = diff / (dist + 1e-9)
                        lp[i]    += direction * push
                        lp[j]    -= direction * push
                        moved     = True
            clamp(lp)
            if not moved:
                break

    repulsion_pass(lpos, max_iter)

    # ── Step 5: fix crossing leader lines by mirroring ────────────────────────
    changed = True
    for _ in range(60):
        if not changed:
            break
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if seg_intersect(pts[i], lpos[i], pts[j], lpos[j]):
                    # Flip the label that is farther from its data point
                    # (smaller disruption)
                    di = np.linalg.norm(lpos[i] - pts[i])
                    dj = np.linalg.norm(lpos[j] - pts[j])
                    if dj >= di:
                        lpos[j] = pts[j] - (lpos[j] - pts[j])
                        clamp(lpos[j:j+1])
                    else:
                        lpos[i] = pts[i] - (lpos[i] - pts[i])
                        clamp(lpos[i:i+1])
                    changed = True

    # ── Step 6: final repulsion after crossing fixes ───────────────────────────
    repulsion_pass(lpos, max_iter)

    # ── Convert back to data coordinates ──────────────────────────────────────
    results = []
    for i in range(n):
        data_xy = Tinv.transform(lpos[i])
        x_dt    = mdates.num2date(data_xy[0]).replace(tzinfo=None)
        y_val   = data_xy[1]
        ha      = "left" if lpos[i][0] >= pts[i][0] else "right"
        results.append((x_dt, y_val, ha))
    return results


saved = apply_config(ax, fig)
if saved:
    label_coords = [(x, y, ha, va) for x, y, ha, va in saved]
else:
    raw = auto_layout(ax, fig, dates, cy, t0,
                      base_dist_px=120, min_sep_px=50,
                      max_iter=1000, repulsion_k=2.0)
    label_coords = [(x, y, ha, "center") for x, y, ha in raw]

for i, name in enumerate(names):
    x_pt, y_pt = dates[i], cy[i]
    x_txt, y_txt, ha, va = label_coords[i]

    ax.annotate("",
                xy=(x_pt, y_pt),
                xytext=(x_txt, y_txt),
                arrowprops=dict(arrowstyle="-", color=LIGHT,
                                linestyle="--", linewidth=2))
    ax.annotate(name,
                xy=(x_pt, y_pt),
                xytext=(x_txt, y_txt),
                fontfamily=MONO, fontsize=18, fontweight="bold",
                ha=ha, va=va,
                annotation_clip=False)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "stat.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(HERE, "stat.pdf"), bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved stat.png / stat.pdf  (total {cumulative[-1]:,} demos, {len(projects)} projects)")
