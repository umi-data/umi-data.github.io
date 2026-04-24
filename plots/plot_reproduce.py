import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 1. Data Preparation (Extracted from image coordinates)
data = [
    (datetime(2024, 3, 1), 2.5, "UMI"),
    (datetime(2024, 6, 20), 3.8, "ManiWav"),
    (datetime(2024, 7, 15), 3.8, "UMI on Legs"),
    (datetime(2024, 10, 15), 13.0, "Fast-UMI"),
    (datetime(2024, 11, 1), 37.0, "Data Scaling\nLaws"),
    (datetime(2024, 11, 20), 38.0, "LEGATO"),
    (datetime(2025, 4, 1), 39.0, "ViTaMIn"),
    (datetime(2025, 5, 15), 48.5, "DexWild"),
    (datetime(2025, 6, 10), 50.0, "DexUMI"),
    (datetime(2025, 7, 20), 53.0, "Touch in the Wild"),
]

dates, values, labels = zip(*data)

# 2. Plotting Configuration
plt.figure(figsize=(10, 8), facecolor='white')
ax = plt.gca()

# Color definitions
primary_blue = "#5469d4"
light_blue = "#b8c2f0"

# 3. Main Plotting
plt.plot(dates, values, color=primary_blue, linewidth=3, zorder=2)
plt.scatter(dates, values, color=primary_blue, s=80, edgecolors=light_blue, linewidths=3, zorder=3)

# 4. Annotations
# We use a mix of right/left alignment based on visual space in the original
alignments = ['right', 'right', 'left', 'left', 'right', 'left', 'left', 'right', 'left', 'right']
offsets = [(-10, 10), (-10, 20), (10, -5), (10, 0), (-10, -10), (10, -20), (10, -15), (-10, -5), (10, -15), (-15, 0)]

for i, txt in enumerate(labels):
    # Draw the dashed lines
    ax.annotate('', xy=(dates[i], values[i]), 
                xytext=(offsets[i][0]*2 + mdates.date2num(dates[i]), values[i] + offsets[i][1]/2),
                arrowprops=dict(arrowstyle='-', color=light_blue, linestyle='--', linewidth=2))
    
    # Draw the text
    plt.annotate(txt, (dates[i], values[i]), 
                 xytext=offsets[i], textcoords='offset points',
                 family='monospace', fontsize=16, fontweight='bold',
                 ha=alignments[i], va='center')

# 5. Styling & Axes
plt.title("Cumulative UMI Demos over Time", fontsize=20, fontweight='bold', family='monospace', pad=20)
plt.ylabel("Thousands of Demos", fontsize=18, fontweight='bold', family='monospace')

# Grid and Spines
plt.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Date Formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
plt.xticks(family='monospace', fontsize=16, fontweight='bold')
plt.yticks(family='monospace', fontsize=16)

plt.tight_layout()
plt.show()