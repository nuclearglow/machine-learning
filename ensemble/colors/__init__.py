#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

cga_vga_r = np.concatenate((np.linspace(85 / 255, 1, 129), np.repeat(1, 127)), axis=0)
cga_vga_g = np.concatenate((np.repeat(1, 127), np.linspace(1, 85 / 255, 129)), axis=0)
cga_vga_ba = np.repeat(1, 256)
cga_vga_colors = np.c_[cga_vga_r, cga_vga_g, cga_vga_ba, cga_vga_ba]
cga_vga = cm.colors.ListedColormap(cga_vga_colors)

cga_0_high_r = np.concatenate((np.linspace(85 / 255, 1, 86), np.repeat(1, 170)), axis=0)
cga_0_high_g = np.concatenate((np.repeat(1, 171), np.linspace(1, 85 / 255, 85)), axis=0)
cga_0_high_b = np.repeat(85 / 255, 256)
cga_0_high_a = np.ones(256)
cga_0_high = cm.colors.ListedColormap(
    np.c_[cga_0_high_r, cga_0_high_g, cga_0_high_b, cga_0_high_a]
)


# * containing colors with linear interpolation
# * along the turning points
# * with resolution times distinct values
def get_colormap(colors, turning_points, resolution):
    
    # Catch an error
    if colors.shape[0] != turning_points.shape[0]:
        return np.nan
    
    colormap = np.zeros((resolution, 3))
    turning_points = np.floor(turning_points * (resolution - 1))

    for i, (s, e) in enumerate(sliding_window(turning_points)):
        start, end = int(s), int(e)
        print(i, start, end)

        start_offset = 0
        index_offset = 0
        if i == 0:
            start_offset = 1
        if i > 0:
            index_offset = 1

        r = np.linspace(
            colors[i, 0], colors[i + 1, 0], int(end - start + 1 + start_offset)
        )
        g = np.linspace(
            colors[i, 1], colors[i + 1, 1], int(end - start + 1 + start_offset)
        )
        b = np.linspace(
            colors[i, 2], colors[i + 1, 2], int(end - start + 1 + start_offset)
        )

        if end != turning_points[-1]:
            r = r[0:-1]
            g = g[0:-1]
            b = b[0:-1]
        else:
            r = r[1:]
            g = g[1:]
            b = b[1:]

        colormap[start + index_offset : end + 1, 0] = r
        colormap[start + index_offset : end + 1, 1] = g
        colormap[start + index_offset : end + 1, 2] = b

    return cm.colors.ListedColormap(colormap)


def sliding_window(iterable, size=2):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


# Palette 1 light
turning_points = np.array([0, 0.25, 0.5, 0.75, 1])
colorlist = np.array(
    [
        [0, 170 / 255, 170 / 255],  # dark cyan
        [85 / 255, 1, 1],  # light cyan
        [170 / 255, 170 / 255, 170 / 255],  # gray
        [1, 85 / 255, 1],  # light magenta
        [170 / 255, 0, 170 / 255],  # dark magenta
    ]
)
cga_p1_light = get_colormap(colorlist, turning_points, 16)

# Palette 1 white
turning_points = np.array([0, 0.25, 0.5, 0.75, 1])
colorlist = np.array(
    [
        [0, 170 / 255, 170 / 255],  # dark cyan
        [85 / 255, 1, 1],  # light cyan
        [1, 1, 1],  # white
        [1, 85 / 255, 1],  # light magenta
        [170 / 255, 0, 170 / 255],  # dark magenta
    ]
)
cga_p1_white = get_colormap(colorlist, turning_points, 16)

# Palette 1 dark
turning_points = np.array([0, 0.25, 0.5, 0.75, 1])
colorlist = np.array(
    [
        [85 / 255, 1, 1],  # light cyan
        [0, 170 / 255, 170 / 255],  # dark cyan
        [0, 0, 0],  # black
        [170 / 255, 0, 170 / 255],  # dark magenta
        [1, 85 / 255, 1],  # light magenta
    ]
)
cga_p1_dark = get_colormap(colorlist, turning_points, 16)

# Palette 3 light
turning_points = np.array([0, 0.25, 0.5, 0.75, 1])
colorlist = np.array(
    [
        [0, 170 / 255, 170 / 255],  # dark cyan
        [85 / 255, 1, 1],  # light cyan
        [170 / 255, 170 / 255, 170 / 255],  # gray
        [1, 85 / 255, 85 / 255],  # light red
        [170 / 255, 0, 0],  # dark red
    ]
)
cga_p3_light = get_colormap(colorlist, turning_points, 16)

# Palette 3 white
turning_points = np.array([0, 0.25, 0.5, 0.75, 1])
colorlist = np.array(
    [
        [0, 170 / 255, 170 / 255],  # dark cyan
        [85 / 255, 1, 1],  # light cyan
        [1, 1, 1],  # white
        [1, 85 / 255, 85 / 255],  # light red
        [170 / 255, 0, 0],  # dark red
    ]
)
cga_p3_white = get_colormap(colorlist, turning_points, 16)

# Palette 3 dark
turning_points = np.array([0, 0.25, 0.5, 0.75, 1])
colorlist = np.array(
    [
        [85 / 255, 1, 1],  # light cyan
        [0, 170 / 255, 170 / 255],  # dark cyan
        [0, 0, 0],  # black
        [170 / 255, 0, 0],  # dark red
        [1, 85 / 255, 85 / 255],  # light red
    ]
)
cga_p3_dark = get_colormap(colorlist, turning_points, 16)

# Tester
def test_cga(cgamap, plotdatfile='testplot.csv'):
    pd = np.flipud(np.loadtxt(open("testplot.csv", "rb"), delimiter=","))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title(f"CGA test")
    cf = ax.imshow(
        pd,
        cmap=cgamap,
        vmin=-np.abs(pd).max(),
        vmax=np.abs(pd).max(),
    )
    fig.colorbar(
        cf,
        boundaries=np.linspace(-np.abs(pd).max(), np.abs(pd).max(), 256),
        orientation="vertical",
    )

test_cga(cga_p1_dark)