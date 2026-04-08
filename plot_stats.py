import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

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
]

projects  = sorted(projects, key=lambda x: x[1])
names     = [p[0] for p in projects]
dates     = [datetime.strptime(p[1], "%Y-%m-%d") for p in projects]
demos     = [p[2] for p in projects]
cumulative = np.cumsum(demos)

t0         = datetime(2024, 1, 15)
plot_dates = [t0] + dates
plot_cum   = [0.0] + [v / 1000 for v in cumulative]


# ── Style constants (from plot_reproduce.py) ──────────────────────────────────
PRIMARY  = "#5469d4"
LIGHT    = "#b8c2f0"
MONO     = "monospace"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
ax.set_facecolor("white")

# ── Slanted line ──────────────────────────────────────────────────────────────
ax.plot(plot_dates, plot_cum,
        color=PRIMARY, linewidth=3, zorder=2, solid_capstyle="round")

# ── Dots ──────────────────────────────────────────────────────────────────────
cx = dates
cy = [v / 1000 for v in cumulative]
ax.scatter(cx, cy, color=PRIMARY, s=80,
           edgecolors=LIGHT, linewidths=3, zorder=3)

# ── Spines ────────────────────────────────────────────────────────────────────
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.xticks(fontfamily=MONO, fontsize=14, fontweight="bold")
plt.yticks(fontfamily=MONO, fontsize=14)

ax.set_ylabel("Thousands of Demos", fontsize=16, fontweight="bold", fontfamily=MONO)
ax.set_title("Cumulative UMI Demos over Time",
             fontsize=20, fontweight="bold", fontfamily=MONO, pad=20)

x_max = dates[-1] + timedelta(days=120)
ax.set_xlim(t0, x_max)
ax.set_ylim(0, max(cumulative) / 1000 * 1.18)

# ── Annotations ───────────────────────────────────────────────────────────────
# (x_day_offset, y_abs_k, ha)
ann_cfg = {
    # early low cluster
    "UMI":               (-25,   6.5, "right"),
    "ManiWav":           (-25,   1.5, "right"),
    "UMI on Legs":       ( 30,  10.5, "left"),
    "Fast-UMI":          ( 30,  15.5, "left"),
    # Oct–Nov 2024 band — push LEGATO far left, DSL right
    "LEGATO":            (-80,  31.0, "right"),
    "Data Scaling Laws": ( 60,  44.0, "left"),
    # ViTaMIn (Apr 25) — isolated x, nudge down
    "ViTaMIn":           ( 30,  35.5, "left"),
    # May 2025 jump — DexWild above line, DexUMI right
    "DexWild":           (-60,  46.5, "right"),
    "DexUMI":            ( 50,  54.0, "left"),
    # Jul 2025
    "Touch in the Wild": ( 50,  49.0, "left"),
    # Sep 2025 cluster — fan out
    "exUMI":             (-60,  57.5, "right"),
    "MV-UMI":            ( 55,  53.0, "left"),
    "ManipForce":        (-60,  63.5, "right"),
    # Oct 2025 big jump
    "FastUMI-100K":      ( 55, 115.0, "left"),
    # Nov 2025 / Feb 2026
    "ViTaMIn-B":         ( 55,  64.0, "left"),
    "HuMI":              ( 40, 158.0, "left"),
}

for i, name in enumerate(names):
    x_pt  = dates[i]
    y_pt  = cumulative[i] / 1000
    xd, y_txt, ha = ann_cfg[name]
    x_txt = x_pt + timedelta(days=xd)

    # dashed leader line
    ax.annotate("",
                xy=(x_pt, y_pt),
                xytext=(x_txt, y_txt),
                arrowprops=dict(
                    arrowstyle="-",
                    color=LIGHT,
                    linestyle="--",
                    linewidth=2,
                ))

    # label text
    ax.annotate(name,
                xy=(x_pt, y_pt),
                xytext=(x_txt, y_txt),
                fontfamily=MONO,
                fontsize=11,
                fontweight="bold",
                ha=ha,
                va="center",
                annotation_clip=False)

plt.tight_layout()
plt.savefig("stat.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved stat.png  (total {cumulative[-1]:,} demos, {len(projects)} projects)")
