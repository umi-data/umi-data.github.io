#!/usr/bin/env python3
"""
interactive_annotate.py  –  Drag annotation labels, adjust arrow direction,
save / load positions.

Controls
--------
  Click label       – select it (turns red)
  Drag label        – move it freely
  Arrow keys        – nudge selected label by 5 px
  Shift+Arrow       – nudge by 1 px
  w                 – upper center  (ha=center, va=bottom)
  a                 – left center   (ha=right,  va=center)
  s                 – bottom center (ha=center, va=top)
  d                 – right center  (ha=left,   va=center)
  Escape            – deselect
  ctrl+s            – save config
  r                 – reset to auto-layout

Buttons
-------
  Save  – write annotation_config.json
  Reset – recompute auto-layout
"""

import os, json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
from datetime import datetime, timedelta
import numpy as np

CONFIG_FILE = "plots/annotation_config.json"

# ── Data ─────────────────────────────────────────────────────────────────────

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
    ("GenRobot",           "2026-04-14", 789772),
    ("Behavior Prompting", "2026-06-29", 1572+5788),
    ("AetheRock",          "2026-06-08", 199+229+497+498+215+212+501+497+496+206+214),
]

projects   = sorted(projects, key=lambda x: x[1])
NAMES      = [p[0] for p in projects]
DATES      = [datetime.strptime(p[1], "%Y-%m-%d") for p in projects]
DEMOS      = [p[2] for p in projects]
CUMULATIVE = np.cumsum(DEMOS)
CY         = CUMULATIVE
T0         = datetime(2024, 1, 15)

PRIMARY = "#5469d4"
LIGHT   = "#b8c2f0"
MONO    = "monospace"
SEL_COL = "#e63946"


# ── Auto-layout (verbatim from plot_stats.py) ─────────────────────────────────
def seg_intersect(a, b, c, d):
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
    """Verbatim from plot_stats.py."""
    fig.canvas.draw()
    T    = ax.transData
    Tinv = T.inverted()
    n    = len(dates)

    pts = np.array([T.transform((mdates.date2num(d), y))
                    for d, y in zip(dates, cy)])

    origin = T.transform((mdates.date2num(t0), 0.0))
    path   = np.vstack([origin, pts])

    bbox = ax.get_window_extent()
    x_lo, x_hi = bbox.x0, bbox.x1
    y_lo, y_hi = bbox.y0, bbox.y1

    normals = np.zeros((n, 2))
    for i in range(n):
        j      = i + 1
        v_in   = path[j] - path[j - 1]
        v_in  /= np.linalg.norm(v_in) + 1e-9
        v_out  = (path[j + 1] - path[j]) if j + 1 < len(path) else v_in
        v_out /= np.linalg.norm(v_out) + 1e-9

        tangent = v_in + v_out
        tlen    = np.linalg.norm(tangent)
        tangent = tangent / tlen if tlen > 1e-9 else v_in

        n_left  = np.array([-tangent[1],  tangent[0]])
        n_right = np.array([ tangent[1], -tangent[0]])

        cross = v_in[0]*v_out[1] - v_in[1]*v_out[0]

        if abs(cross) < 0.05:
            candidate = n_left if n_left[1] > n_right[1] else n_right
        else:
            candidate = n_right if cross > 0 else n_left

        normals[i] = candidate

    lpos = pts + normals * base_dist_px

    MARGIN = 8
    def clamp(lp):
        lp[:, 0] = np.clip(lp[:, 0], x_lo + MARGIN, x_hi - MARGIN)
        lp[:, 1] = np.clip(lp[:, 1], y_lo + MARGIN, y_hi - MARGIN)

    clamp(lpos)

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

    changed = True
    for _ in range(60):
        if not changed:
            break
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if seg_intersect(pts[i], lpos[i], pts[j], lpos[j]):
                    di = np.linalg.norm(lpos[i] - pts[i])
                    dj = np.linalg.norm(lpos[j] - pts[j])
                    if dj >= di:
                        lpos[j] = pts[j] - (lpos[j] - pts[j])
                        clamp(lpos[j:j+1])
                    else:
                        lpos[i] = pts[i] - (lpos[i] - pts[i])
                        clamp(lpos[i:i+1])
                    changed = True

    repulsion_pass(lpos, max_iter)

    results = []
    for i in range(n):
        data_xy = Tinv.transform(lpos[i])
        x_dt    = mdates.num2date(data_xy[0]).replace(tzinfo=None)
        y_val   = data_xy[1]
        ha      = "left" if lpos[i][0] >= pts[i][0] else "right"
        results.append((x_dt, y_val, ha))
    return results


# ── Editor ────────────────────────────────────────────────────────────────────
class AnnotationEditor:

    def __init__(self):
        self.selected = None
        self.dragging = False
        self.drag_offset = (0.0, 0.0)   # display-px offset (text pos – click)
        self._ann_lines  = []            # arrow Annotation objects
        self._ann_texts  = []            # text  Annotation objects
        self.config      = []            # list of dicts per label

        self._build_figure()
        self._load_or_reset()
        self._redraw()
        self._connect()
        plt.show()

    # ── Figure skeleton ───────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 9), facecolor="white")
        self.ax.set_facecolor("white")

        plot_dates = DATES
        plot_cum   = [v / 1000 for v in CY]
        self.ax.plot(plot_dates, plot_cum, color=PRIMARY, linewidth=3,
                     zorder=2, solid_capstyle="round")
        self.ax.scatter(DATES, CY, color=PRIMARY, s=80,
                        edgecolors=LIGHT, linewidths=3, zorder=3)

        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linewidth(2)
        self.ax.spines["bottom"].set_linewidth(2)
        self.ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
        self.ax.set_axisbelow(True)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        self.ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        plt.xticks(fontfamily=MONO, fontsize=18, fontweight="bold")
        plt.yticks(fontfamily=MONO, fontsize=18)
        self.ax.set_ylabel("Number of Demos", fontsize=22,
                           fontweight="bold", fontfamily=MONO)
        self.ax.set_title("Cumulative UMI Demos over Time",
                          fontsize=28, fontweight="bold", fontfamily=MONO, pad=20)
        x_max = DATES[-1] + timedelta(days=120)
        self.ax.set_yscale("log")
        self.ax.set_xlim(T0, x_max)
        self.ax.set_ylim(min(plot_cum) * 0.5, max(CUMULATIVE) * 2.0 / 1000)

        # Buttons
        self.fig.subplots_adjust(bottom=0.09)
        for spine in ['top', 'right', 'left', 'bottom']:
            pass

        ax_save  = self.fig.add_axes([0.72, 0.015, 0.10, 0.045])
        ax_reset = self.fig.add_axes([0.84, 0.015, 0.10, 0.045])
        self.btn_save  = Button(ax_save,  'Save (^s)')
        self.btn_reset = Button(ax_reset, 'Reset (r)')
        self.btn_save .on_clicked(lambda _: self._save())
        self.btn_reset.on_clicked(lambda _: self._reset())

        self._status = self.fig.text(
            0.01, 0.015,
            "Click a label to select  |  drag to move  |  arrow keys: nudge  "
            "|  waxd: align (up/left/down/right)  |  ctrl+s: save",
            fontsize=8, fontfamily=MONO, va='bottom', color="#555555")

    # ── Config helpers ────────────────────────────────────────────────────────
    def _load_or_reset(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    raw = json.load(f)
                # Build a name→entry map so order / additions are robust
                by_name = {e['name']: e for e in raw}
                self.config = []
                for name in NAMES:
                    if name in by_name:
                        e = by_name[name]
                        self.config.append({
                            'name':  name,
                            'x_txt': datetime.strptime(e['x_txt'], "%Y-%m-%dT%H:%M:%S"),
                            'y_txt': float(e['y_txt']),
                            'ha':    e['ha'],
                            'va':    e['va'],
                        })
                    else:
                        self.config.append(None)   # will be filled by auto-layout
                # Fill any missing entries via auto-layout
                missing = [i for i, c in enumerate(self.config) if c is None]
                if missing:
                    self._fill_auto(missing)
                self._status_msg(f"Loaded {CONFIG_FILE}")
                return
            except Exception as exc:
                print(f"[warn] could not load config: {exc}")
        self._reset()

    def _clamp_config_to_canvas(self):
        """Push every text anchor inside the axes area (with padding for text width)."""
        self.fig.canvas.draw()
        T    = self.ax.transData
        Tinv = T.inverted()
        bbox = self.ax.get_window_extent()
        PAD  = 70   # px — generous so the text box itself stays fully visible
        x_lo = bbox.x0 + PAD
        x_hi = bbox.x1 - PAD
        y_lo = bbox.y0 + PAD
        y_hi = bbox.y1 - PAD
        for c in self.config:
            dp = T.transform((mdates.date2num(c['x_txt']), c['y_txt'])).copy()
            dp[0] = float(np.clip(dp[0], x_lo, x_hi))
            dp[1] = float(np.clip(dp[1], y_lo, y_hi))
            data = Tinv.transform(dp)
            c['x_txt'] = mdates.num2date(data[0]).replace(tzinfo=None)
            c['y_txt'] = float(data[1])

    def _fill_auto(self, indices):
        coords = auto_layout(self.ax, self.fig, DATES, CY, T0,
                             base_dist_px=120, min_sep_px=50,
                             max_iter=1000, repulsion_k=2.0)  # matches plot_stats.py
        for i in indices:
            x_txt, y_txt, ha = coords[i]
            self.config[i] = {'name': NAMES[i], 'x_txt': x_txt,
                              'y_txt': y_txt, 'ha': ha, 'va': 'center'}
        self._clamp_config_to_canvas()

    def _reset(self):
        coords = auto_layout(self.ax, self.fig, DATES, CY, T0,
                             base_dist_px=120, min_sep_px=50,
                             max_iter=1000, repulsion_k=2.0)  # matches plot_stats.py
        self.config = []
        for i, name in enumerate(NAMES):
            x_txt, y_txt, ha = coords[i]
            self.config.append({'name': name, 'x_txt': x_txt,
                                'y_txt': y_txt, 'ha': ha, 'va': 'center'})
        self._clamp_config_to_canvas()
        self.selected = None
        self._redraw()
        self._status_msg("Reset to auto-layout")

    def _save(self):
        out = []
        for c in self.config:
            out.append({
                'name':  c['name'],
                'x_txt': c['x_txt'].strftime("%Y-%m-%dT%H:%M:%S"),
                'y_txt': c['y_txt'],
                'ha':    c['ha'],
                'va':    c['va'],
            })
        with open(CONFIG_FILE, 'w') as f:
            json.dump(out, f, indent=2)
        self._status_msg(f"Saved → {CONFIG_FILE}")

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _redraw(self):
        for art in self._ann_lines + self._ann_texts:
            try: art.remove()
            except Exception: pass
        self._ann_lines.clear()
        self._ann_texts.clear()

        for i, name in enumerate(NAMES):
            c        = self.config[i]
            x_pt     = DATES[i]
            y_pt     = CY[i]
            x_txt    = c['x_txt']
            y_txt    = c['y_txt']
            ha, va   = c['ha'], c['va']
            selected = (i == self.selected)
            color    = SEL_COL if selected else PRIMARY

            line = self.ax.annotate(
                "", xy=(x_pt, y_pt), xytext=(x_txt, y_txt),
                arrowprops=dict(arrowstyle="-",
                                color=SEL_COL if selected else LIGHT,
                                linestyle="--", linewidth=2.5 if selected else 2),
                annotation_clip=False)

            kw = dict(fontfamily=MONO, fontsize=18, fontweight="bold",
                      ha=ha, va=va, color=color, annotation_clip=False)
            if selected:
                kw['bbox'] = dict(boxstyle='round,pad=0.25',
                                  facecolor='#fff9f9', edgecolor=SEL_COL,
                                  linewidth=1.2, alpha=0.9)
            txt = self.ax.annotate(name, xy=(x_pt, y_pt),
                                   xytext=(x_txt, y_txt), **kw)

            self._ann_lines.append(line)
            self._ann_texts.append(txt)

        self.fig.canvas.draw_idle()

    # ── Event helpers ─────────────────────────────────────────────────────────
    def _display_pos(self, i):
        """Display-pixel position of annotation i's text anchor."""
        c = self.config[i]
        return self.ax.transData.transform(
            (mdates.date2num(c['x_txt']), c['y_txt']))

    def _set_data_pos(self, i, disp_xy):
        """Set config[i] text position from display-pixel coords."""
        data = self.ax.transData.inverted().transform(disp_xy)
        self.config[i]['x_txt'] = mdates.num2date(data[0]).replace(tzinfo=None)
        self.config[i]['y_txt'] = float(data[1])

    def _status_msg(self, msg):
        self._status.set_text(msg)
        self.fig.canvas.draw_idle()

    # ── Events ────────────────────────────────────────────────────────────────
    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect('button_press_event',   self._on_press)
        c.mpl_connect('motion_notify_event',  self._on_motion)
        c.mpl_connect('button_release_event', self._on_release)
        c.mpl_connect('key_press_event',      self._on_key)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self.fig.canvas.draw()   # ensure renderer is current

        hit = None
        for i, txt in enumerate(self._ann_texts):
            bb = txt.get_window_extent()
            if bb.contains(event.x, event.y):
                hit = i
                break

        if hit is not None:
            self.selected = hit
            self.dragging = True
            dp = self._display_pos(hit)
            self.drag_offset = (dp[0] - event.x, dp[1] - event.y)
            name = NAMES[hit]
            c    = self.config[hit]
            self._status_msg(
                f"[{name}]  ha={c['ha']}  va={c['va']}  "
                "| waxd: align up/left/down/right  | arrows: nudge  | ctrl+s: save")
        else:
            self.selected = None
            self.dragging = False
            self._status_msg(
                "Click a label to select  |  drag  |  arrow keys: nudge  "
                "|  waxd: align  |  ctrl+s: save")
        self._redraw()

    def _on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        new_disp = (event.x + self.drag_offset[0],
                    event.y + self.drag_offset[1])
        self._set_data_pos(self.selected, new_disp)
        self._redraw()

    def _on_release(self, event):
        self.dragging = False

    def _on_key(self, event):
        if event.key == 'escape':
            self.selected = None
            self._redraw()
            return

        if event.key == 'ctrl+s':
            self._save()
            return

        if event.key == 'r':
            self._reset()
            return

        if self.selected is None:
            return

        i   = self.selected
        c   = self.config[i]
        key = event.key

        # ── Arrow keys: nudge text position ───────────────────────────────
        step = 1 if 'shift' in key else 5
        key  = key.replace('shift+', '')
        dp   = self._display_pos(i).copy()
        if   key == 'right': dp[0] += step
        elif key == 'left':  dp[0] -= step
        elif key == 'up':    dp[1] += step
        elif key == 'down':  dp[1] -= step

        # ── waxd: preset alignments ───────────────────────────────────────
        elif key == 'w':  c['ha'] = 'center'; c['va'] = 'top'  # upper center
        elif key == 'a':  c['ha'] = 'left';  c['va'] = 'center'  # left center
        elif key == 'x':  c['ha'] = 'center'; c['va'] = 'bottom'     # bottom center
        elif key == 'd':  c['ha'] = 'right';   c['va'] = 'center'  # right center
        else: return

        if key in ('right', 'left', 'up', 'down'):
            self._set_data_pos(i, dp)

        self._status_msg(
            f"[{NAMES[i]}]  ha={c['ha']}  va={c['va']}  "
            "| waxd: align up/left/down/right  | arrows: nudge  | ctrl+s: save")
        self._redraw()


# ── Utility: apply saved config to plot_stats figure ─────────────────────────
def apply_config(ax, fig, config_file=CONFIG_FILE):
    """
    Call this from plot_stats.py instead of auto_layout to use saved positions.
    Returns list of (x_txt_datetime, y_txt_float, ha_str, va_str) or None if
    no config file is found.
    """
    if not os.path.exists(config_file):
        return None
    with open(config_file) as f:
        raw = json.load(f)
    by_name = {e['name']: e for e in raw}
    result  = []
    for name in NAMES:
        if name not in by_name:
            return None   # incomplete config – let caller fall back to auto
        e = by_name[name]
        result.append((
            datetime.strptime(e['x_txt'], "%Y-%m-%dT%H:%M:%S"),
            float(e['y_txt']),
            e['ha'],
            e.get('va', 'center'),
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    AnnotationEditor()
