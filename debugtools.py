from __future__ import division, absolute_import, print_function
import numpy as np
from svgpathtools import Line, disvg, svg2paths


def normalvf(paths, cols, l=None, sw=.1):
    """This will display the normal vector field for the collections of curves,
    `paths` (and red disks on singular points), which can be useful when
    checking an svg for issues with orientation, smoothness, curvature, etc."""
    if not l:
        l = min(p.length() for p in paths) / 20
        print('l = ', l)
    normals = []
    singular_pts = []
    for p in paths:
        for t in np.linspace(0, 1, 100):
            try:
                normals.append(Line(p.point(t), p.point(t) + l * p.normal(t)))
            except:
                singular_pts.append(p.point(t))
    paths2disp = paths + normals
    try:
        colors = [att['stroke'] for att in cols] + ['purple'] * len(normals)
    except:
        colors = cols + ['purple'] * len(normals)
    disvg(paths2disp, colors, nodes=singular_pts,
          stroke_widths=[sw] * len(paths2disp))


def svg2normalvf(fn, l=None, sw=.1):
    paths, atts = svg2paths(fn)
    normalvf(paths, atts, l=l, sw=sw)