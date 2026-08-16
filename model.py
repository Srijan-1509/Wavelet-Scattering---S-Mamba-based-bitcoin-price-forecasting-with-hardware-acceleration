import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(-0.8, 4.8)
ax.set_ylim(-1.2, 5.2)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ── 4 cols x 5 rows = 20 nodes ────────────────────────────────
nodes = {}
for row in range(5):
    for col in range(4):
        node_id = row * 4 + col
        nodes[node_id] = (col, 4 - row)

hub_nodes_0 = [0, 7]
hub_nodes_1 = [17, 18]

# ── Horizontal wired links ─────────────────────────────────────
for row in range(5):
    for col in range(3):
        nid = row * 4 + col
        x1, y1 = nodes[nid]
        x2, y2 = nodes[nid + 1]
        ax.plot([x1, x2], [y1, y2], '-', color='#2c3e50',
                lw=2.0, zorder=1)

# ── Butterfly cross-connections ────────────────────────────────
butterfly_connections = [
    (0,1,0,0),(0,1,0,2),(0,1,1,1),(0,1,1,3),
    (0,1,2,0),(0,1,2,2),(0,1,3,1),(0,1,3,3),
    (1,2,0,0),(1,2,0,1),(1,2,1,0),(1,2,1,1),
    (1,2,2,2),(1,2,2,3),(1,2,3,2),(1,2,3,3),
    (2,3,0,0),(2,3,0,2),(2,3,1,1),(2,3,1,3),
    (2,3,2,0),(2,3,2,2),(2,3,3,1),(2,3,3,3),
    (3,4,0,0),(3,4,0,1),(3,4,1,0),(3,4,1,1),
    (3,4,2,2),(3,4,2,3),(3,4,3,2),(3,4,3,3),
]

for (rt, rb, ct, cb) in butterfly_connections:
    n_top = rt * 4 + ct
    n_bot = rb * 4 + cb
    x1, y1 = nodes[n_top]
    x2, y2 = nodes[n_bot]
    lc = '#2980b9' if ct == cb else '#27ae60'
    ax.plot([x1, x2], [y1, y2], '-', color=lc,
            lw=1.6, zorder=1, alpha=0.65)

# ── Hub positions (outside grid) ──────────────────────────────
# Hub 0: to the right of the grid (near nodes 0 and 7)
# Hub 1: below the grid (near nodes 17 and 18)
hub0_pos = (4.5, 2.0)   # right side, vertically between R0 and R7
hub1_pos = (1.5, -0.8)  # bottom, horizontally between R17 and R18

# ── Wired node-to-hub connections ─────────────────────────────
for nid in hub_nodes_0:
    x, y = nodes[nid]
    hx, hy = hub0_pos
    ax.plot([x, hx], [y, hy], '-', color='#e74c3c',
            lw=2.0, zorder=2, alpha=0.8)

for nid in hub_nodes_1:
    x, y = nodes[nid]
    hx, hy = hub1_pos
    ax.plot([x, hx], [y, hy], '-', color='#8e44ad',
            lw=2.0, zorder=2, alpha=0.8)

# ── Wireless link: Hub0 <-> Hub1 (main wireless link) ─────────
def draw_wireless(ax, p1, p2, color, label_txt=None):
    x1, y1 = p1
    x2, y2 = p2
    mx, my = (x1+x2)/2, (y1+y2)/2
    offset_x = -(y2-y1)*0.25
    offset_y =  (x2-x1)*0.25
    cx, cy = mx+offset_x, my+offset_y
    t = np.linspace(0, 1, 300)
    bx = (1-t)**2*x1 + 2*(1-t)*t*cx + t**2*x2
    by = (1-t)**2*y1 + 2*(1-t)*t*cy + t**2*y2
    ax.plot(bx, by, '--', color=color, lw=2.8,
            dashes=(9, 4), zorder=3, alpha=0.95)
    # Arrow at midpoint
    mid = len(t)//2
    ax.annotate('', xy=(bx[mid+6], by[mid+6]),
                xytext=(bx[mid-6], by[mid-6]),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.2))
    ax.annotate('', xy=(bx[mid-6], by[mid-6]),
                xytext=(bx[mid+6], by[mid+6]),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.2))
    if label_txt:
        ax.text(cx+0.1, cy+0.1, label_txt,
                fontsize=8.5, color=color, fontweight='bold',
                ha='center', va='center',
                bbox=dict(fc='white', ec=color, boxstyle='round,pad=0.2',
                          alpha=0.9))

# Main wireless link between hubs
draw_wireless(ax, hub0_pos, hub1_pos, '#c0392b',
              label_txt='60 GHz\nWireless')

# ── Also show all effective wireless paths (node-level) ────────
# All 4 combinations: 0↔17, 0↔18, 7↔17, 7↔18 (faded)
for n0 in hub_nodes_0:
    for n1 in hub_nodes_1:
        x1, y1 = nodes[n0]
        x2, y2 = nodes[n1]
        ax.plot([x1, x2], [y1, y2], ':',
                color='#e74c3c', lw=1.2, zorder=1,
                alpha=0.25, dashes=(3, 5))

# ── Draw Hub symbols (diamond shape) ──────────────────────────
def draw_hub(ax, pos, color, label):
    hx, hy = pos
    size = 0.28
    diamond = plt.Polygon(
        [[hx, hy+size], [hx+size, hy],
         [hx, hy-size], [hx-size, hy]],
        closed=True, fc=color, ec='black',
        lw=2.0, zorder=5, alpha=0.92)
    ax.add_patch(diamond)
    ax.text(hx, hy, 'H', ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white', zorder=6)
    ax.text(hx, hy-size-0.18, label,
            ha='center', va='top', fontsize=9,
            fontweight='bold', color=color,
            bbox=dict(fc='white', ec=color,
                      boxstyle='round,pad=0.25',
                      alpha=0.95))

draw_hub(ax, hub0_pos, '#e74c3c', 'Hub 0')
draw_hub(ax, hub1_pos, '#8e44ad', 'Hub 1')

# ── Router squares ─────────────────────────────────────────────
for nid, (x, y) in nodes.items():
    ax.add_patch(FancyBboxPatch(
        (x-0.27, y-0.27), 0.54, 0.54,
        boxstyle="round,pad=0.04",
        fc='#dde3ea', ec='#7f8c8d',
        lw=1.2, zorder=2))

# ── Node circles ───────────────────────────────────────────────
for nid, (x, y) in nodes.items():
    if nid in hub_nodes_0:
        fc='#f9c0bb'; ec='#e74c3c'; lw=2.8; r=0.19
    elif nid in hub_nodes_1:
        fc='#d7bff5'; ec='#8e44ad'; lw=2.8; r=0.19
    else:
        fc='#eaf2fb'; ec='#2c3e50'; lw=1.5; r=0.16

    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec,
                             lw=lw, zorder=3))
    # Node label INSIDE circle
    ax.text(x, y, f'R{nid}', ha='center', va='center',
            fontsize=7, fontweight='bold',
            color='#1a1a1a', zorder=5)

# ── Annotate hub-connected nodes ──────────────────────────────
for nid in hub_nodes_0:
    x, y = nodes[nid]
    ax.text(x, y+0.32, f'→Hub0', ha='center', va='bottom',
            fontsize=7, color='#e74c3c', fontstyle='italic')

for nid in hub_nodes_1:
    x, y = nodes[nid]
    ax.text(x, y-0.32, f'→Hub1', ha='center', va='top',
            fontsize=7, color='#8e44ad', fontstyle='italic')

# ── Legend ─────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(fc='#f9c0bb', ec='#e74c3c', lw=2,
                   label='Hub 0 nodes (R0, R7)'),
    mpatches.Patch(fc='#d7bff5', ec='#8e44ad', lw=2,
                   label='Hub 1 nodes (R17, R18)'),
    mpatches.Patch(fc='#eaf2fb', ec='#2c3e50', lw=1.5,
                   label='Standard router'),
    mpatches.Patch(fc='#e74c3c', ec='black', lw=1.5,
                   label='Wireless hub (diamond)'),
    plt.Line2D([0],[0], color='#2c3e50', lw=2,
               label='Horizontal wired link'),
    plt.Line2D([0],[0], color='#2980b9', lw=2,
               label='Vertical butterfly link'),
    plt.Line2D([0],[0], color='#27ae60', lw=2,
               label='Diagonal butterfly cross-link'),
    plt.Line2D([0],[0], color='#e74c3c', lw=2,
               label='Node-to-hub wired link'),
    plt.Line2D([0],[0], color='#c0392b', lw=2.5,
               ls='--', dashes=(9,4),
               label='Hub-to-hub wireless (60 GHz)'),
    plt.Line2D([0],[0], color='#e74c3c', lw=1.2,
               ls=':', alpha=0.5,
               label='Effective wireless paths (R0/R7 ↔ R17/R18)'),
]
ax.legend(handles=legend_elements,
          loc='upper left',
          fontsize=8, framealpha=0.97,
          edgecolor='#bdc3c7',
          bbox_to_anchor=(-0.75, 1.02))

ax.set_title('Butterfly WiNoC Topology — 4×5 Mesh with Wireless Hubs\n'
             'DFT-S-OFDM PHY Layer Integration (Noxim Simulation)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('winoc_butterfly_topology.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print("Saved: winoc_butterfly_topology.png")