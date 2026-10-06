"""
Scale bars for image panels.

Pixel sizes of the image tiles used in the paper figures:
    PMD (WSI): 0.46 um/pixel
    STP:       0.35 um/pixel
"""

import matplotlib.patheffects as patheffects
from matplotlib.patches import Rectangle

UM_PER_PX = {
    'pmd': 0.46,
    'stp': 0.35,
}

# Bar lengths to choose from (um)
NICE_LENGTHS_UM = [5, 10, 20, 50, 100, 200, 500, 1000]


def pick_length_um(panel_width_px, um_per_px, max_fraction=0.25):
    """Largest standard length that fits in max_fraction of the panel width."""
    limit_um = panel_width_px * um_per_px * max_fraction
    fitting = [l for l in NICE_LENGTHS_UM if l <= limit_um]
    return fitting[-1] if fitting else NICE_LENGTHS_UM[0]


def add_scale_bar(ax, panel_width_px, panel_height_px, um_per_px,
                  length_um=None, fontsize=10, label=True):
    """
    Draw a white scale bar in the bottom-right corner of an imshow panel.

    The axes must be in pixel (data) coordinates, as set up by imshow.
    Returns the bar length in um so callers can state it in the legend.
    """
    if length_um is None:
        length_um = pick_length_um(panel_width_px, um_per_px)
    length_px = length_um / um_per_px

    margin = 0.05 * panel_width_px
    thickness = max(2.0, 0.015 * panel_height_px)
    x0 = panel_width_px - margin - length_px
    y0 = panel_height_px - margin - thickness

    ax.add_patch(Rectangle((x0, y0), length_px, thickness,
                           facecolor='white', edgecolor='black', linewidth=0.5))
    if label:
        text = ax.text(x0 + length_px / 2, y0 - 0.01 * panel_height_px, f'{length_um} µm',
                       color='white', fontsize=fontsize, ha='center', va='bottom')
        text.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='black')])
    return length_um
