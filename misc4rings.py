# External Dependencies
from __future__ import division
from scipy.integrate import quad
from math import sqrt, cos, sin, pi
from cmath import phase
from operator import itemgetter
from os import getcwd, path as os_path
from andysmod import memoize, bool2bin, eucdist_numpy
from svgpathtools import Line, CubicBezier, Path, parse_path, wsvg, disvg

# Internal Dependencies
import options4rings as opt
from andysSVGpathTools import (segDerivative, extremePointInPath, path2str,
                               closestPointInPath,
                               pathXpathIntersections as
                               andysSVGpathTools_pathXpathIntersections)


def inv_arclength(curve, s):
    return curve.ilength(s)


def sortby(x, k):
    return sorted(x, key=itemgetter(k))


def pasteableRing(ring):
    s = "Ring('%s', '%s', '%s', Radius(%s), parse_path('%s'))" \
        "" % (ring.string, ring.color, ring.brook_tag, ring.center, ring.string)
    return s


def isApproxClosedPath(path):
    """takes in path object and outputs true if closed (within tolerance)"""
    return abs(path[0].start - path[-1].end) < opt.tol_isApproxClosedPath


def isNotTooFarFrom(p, q):
    """takes two complex numbers, returns boolean"""
    tol_isNotTooFarFrom = 10
    return abs(p-q) < tol_isNotTooFarFrom


def isNear(p, q):
    """takes two complex numbers, returns boolean"""
    return abs(p-q) < opt.tol_isNear


def isDegenerateSegment(seg, tol_degSeg=1):  # #### TOLERANCE
    return abs(seg.start - seg.end) < tol_degSeg


def isCCW(path, center):

    # note: phase range is [-pi,pi] (not continuous on negative real axis)
    theta = lambda t: phase(path.point(t) - center)

    n = 100
    s = sum(bool2bin(theta(float(k+1)/n) - theta(float(k)/n) > 0) for k in range(n))
    if s > .7 * n:
        return True
    elif s < .3 * n:
        return False
    else:
        raise Exception("Either something is wrong with isCCW, or this path "
                        "is far from being a function of theta.")


def sortRingListByMinR(ring_list):
    return [val for (key, val) in sorted(
        [(ring.minR, ring) for ring in ring_list])]


from andysSVGpathTools import areaEnclosed as andysSVGpathTools_areaEnclosed
@memoize
def areaEnclosed(path):
    return andysSVGpathTools_areaEnclosed(path)


def aveRadiusEnclosed(path, center):
    """UNFINISHED.
    returns absolute value of area enclosed by path note: negative area
    results from CW (as oppossed to CCW) parameterization of path"""
    if not isApproxClosedPath(path):
        raise Exception('Path encloses no area.')
    #WHOOPS... need to redo integrand as I'm using stoke's
    raise Exception("This fcn is not yet implemented")
#    rad = lambda t: sqrt((x(t)-center.real)**2+(y(t)-center.imag)**2)
    aveEnclosed = 0
    aveEnclosed_error = 0
    for seg in path:
        if isinstance(seg,CubicBezier):
            p = (seg.start, seg.control1, seg.control2, seg.end)
            a_0, a_1, a_2, a_3 = [z.real for z in p]
            b_0, b_1, b_2, b_3 = [z.imag for z in p]

            # real part of cubic Bezier
            x = lambda t: \
                a_0*(1-t)**3 + 3*a_1*t*(1-t)**2 + 3*a_2*(1-t)*t**2 + a_3*t**3

            # imag part of cubic Bezier
            y = lambda t: \
                b_0*(1-t)**3 + 3*b_1*t*(1-t)**2 + 3*b_2*(1-t)*t**2 + b_3*t**3

            # derivative of imaginary part of cubic Bezier
            dy = lambda t: \
                3*(b_1-b_0)*(1-t)**2 + 6*(b_2-b_1)*t*(1-t) + 3*(b_3-b_2)*t**2

            integrand = lambda t: x(t)*dy(t)

        elif isinstance(seg, Line):
            x = lambda t: (1 - t)*seg.start.real + t*seg.end.real
            dy = seg.end.imag - seg.start.imag
            integrand = lambda t: sqrt(x(t)**2 + y(t)**2) * x(t) * dy
        else:
            raise Exception(
                'Path segment is neither Line object nor CubicBezier object.')
        result = quad(integrand, 0, 1)
        aveEnclosed += result[0]
        aveEnclosed_error += result[1]

    # #### Tolerance should be based on native res of SVG
    # (what is area of single pixel)
    if aveEnclosed_error > 1e-2:
        raise Exception('area may be erroneous')
    return abs(aveEnclosed/areaEnclosed(path))


def aveRadius_path(path, origin):
    """

    Args:
        path:
        origin: complex number

    Returns:
        the average distance between origin and path

    """
    (aveRad,aveRad_error) = (0, 0)
    for seg in path:
        if isinstance(seg, CubicBezier):
            p_0, p_1, p_2, p_3 = seg.start, seg.control1, seg.control2, seg.end

            z = lambda t: p_0*(1-t)**3 + 3*p_1*t*(1-t)**2 + 3*p_2*(1-t)*t**2 + p_3*t**3
            dz = lambda t: 3*(p_1-p_0)*(1-t)**2 + 6*(p_2-p_1)*t*(1-t) + 3*(p_3-p_2)*t**2

            integrand1 = lambda t: abs((z(t)-origin)*dz(t))
            integrand2 = lambda t: abs(dz(t))

            # could also use seg.lenth() instead of integrating integrand2
            result = quad(integrand1, 0, 1) / quad(integrand2, 0, 1)
        elif isinstance(seg, Line):
            z = lambda t: (1 - t)*seg.start + t*seg.end
            dz = seg.end-seg.start
            integrand1 = lambda t: abs((z(t)-origin)*dz)
            result = quad(integrand1, 0, 1) / abs(dz)
        else:
            raise Exception(
                'Path segment is neither Line object nor CubicBezier object.')

        aveRad += result[0]
        aveRad_error += result[1]

    # Tolerance should be based on native res of SVG
    # (what is area of single pixel)
    if aveRad_error > 10**(-4):
        raise Exception('area may be erroneous')
    return abs(aveRad)


def pathXpathIntersections(path1, path2, justonemode=False):
    """this fcn is a workaround needed as memoize doesn't like keyword
    arguments"""
    return pathXpathIntersections_thefunction(path1, path2, justonemode)


@memoize
def pathXpathIntersections_thefunction(path1, path2, justonemode):
    """this fcn is a workaround needed as memoize doesn't like keyword
    arguments"""
    return andysSVGpathTools_pathXpathIntersections(path1, path2,
                                                    justonemode=justonemode)


def centerSquare(c):
    tr = c+1+1j
    tl = c-1+1j
    br = c+1-1j
    bl = c-1-1j
    d = "M %s,%s L%s,%s L%s,%s L%s,%sz" \
        "" % (bl.real, bl.imag,
              tl.real, tl.imag,
              tr.real, tr.imag,
              br.real, br.imag)
    return d


def hex2rgb(value):
    # from https://stackoverflow.com/questions/214359
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv//3], 16) for i in range(0, lv, lv//3))


def rgb2hex(rgb):
    # from https://stackoverflow.com/questions/214359
    hcolor = '#%02x%02x%02x'%rgb
    return hcolor.upper()


def argmin(somelist):
    return min(enumerate(somelist), key=itemgetter(1))


def closestColor(hexcolor, colordict):
    color = hex2rgb(hexcolor)
    rgbPalette = [hex2rgb(c) for c in colordict.values()]
    distances = [eucdist_numpy(c, color) for c in rgbPalette]
    return rgb2hex(rgbPalette[argmin(distances)[0]])


def path_isbelow_point(path, pt, center):
    line2center = Line(pt, center)
    if pathXpathIntersections(path, line2center):
        return True
    else:
        return False


def normalLineAtT_toInner_intersects_withOuter(
        innerT, inner_path, outer_path, center, debug=False, testlength=None):
    """ Finds normal line to `inner_path` at `innerT`

    Note: This fcn is meant to be used by the
    `IncompleteRing.findTransects2endpointsFromInnerPath_normal` method

    Returns:
        (normal_Line, outer_seg, outer_t) where normal_Line is the Line
            object normal to `inner_path` beginning at
            `inner_path.point(innerT)` and ending at `outer_seg.point(outer_t)`
            or, in the case that the normal line never intersects `outer_path`,
            returns (False, False, False).
    """
    assert 0 <= innerT <= 1
    if isApproxClosedPath(inner_path) and innerT == 1:
        innerT = 0

    if innerT == 0:
        inner_pt = inner_path[0].start
    elif innerT == 1:
        inner_pt = inner_path[-1].end
    else:
        inner_pt = inner_path.point(innerT)

    try:
        n_vec = inner_path.normal(innerT)
        if abs(n_vec) == 0:
            raise
    except:
        # This is an adhoc fix for when (?precision?) issues cause nvec to be
        # ill-defined
        sgn = -1 if innerT > 0.5 else 1
        dt = sgn * 0.001
        k = 0
        while k < 100:
            k += 1
            offset_T = innerT + k*dt
            try:
                n_vec = inner_path.normal(offset_T)
                break
            except:
                pass
        if k >= 100:
            raise Exception("Problem finding normal vector.")

    if testlength:
        long_mag = testlength
    else:
        # find the max length to the outer
        long_mag = 1.1 * outer_path.radialrange(inner_pt)[1][0]
    long_norm_line = Line(inner_pt, inner_pt + long_mag*n_vec)
    long_norm_path = Path(long_norm_line)
    intersec = sortby(pathXpathIntersections(long_norm_path, outer_path), 2)
    if intersec:
        # this tests if the normal line intersect with the inner (other
        # than at its startpoint) before hitting the outer
        # note: pathxpath returns (seg1,seg2,t1,t2)
        intersec_withInner = \
            sortby(pathXpathIntersections(Path(long_norm_line), inner_path), 2)
        if len(intersec_withInner) > 1 and intersec_withInner[1][2] < intersec[0][2]:
            if debug:
                return long_norm_line, False, False
            else:
                return False, False, False

        # this block tests if the normal line goes through the center (roughly
        # speaking) and finds a false intersection with outer on the other
        # side of the cross-section
        path_is_a_line = (len(inner_path) == 1 and isinstance(inner_path[0], Line))
        if not inner_path.isclosed() and not path_is_a_line:
            innerClosingLine = Line(inner_path[0].start, inner_path[-1].end)
            closingLine_self_intersections = \
                pathXpathIntersections(inner_path, Path(innerClosingLine))
            bad_close = False
            for inters in closingLine_self_intersections:
                pt = inters[0].point(inters[2])
                if not (isNear(pt, inner_path[0].start) or
                        isNear(pt, inner_path[-1].end)):
                    bad_close = True
            if not bad_close:
                intersec_withClosingLine = \
                    sortby(pathXpathIntersections(Path(long_norm_line),
                                                  Path(innerClosingLine)), 2)

                if (intersec_withClosingLine and
                        intersec_withClosingLine[0][2] < intersec[0][2] and not
                        isNear(inner_pt,
                               intersec_withClosingLine[0][1].point(
                                   intersec_withClosingLine[0][3]))):
                    if debug:
                        return long_norm_line, False, False
                    else:
                        return False, False, False
            #UHHHwhatifbadclose???
        seg_nl, seg_out, t_nl, t_out = intersec[0]
        nlin = Line(inner_pt, seg_out.point(t_out))
        return nlin, seg_out, t_out
    else:
        if debug:
            return long_norm_line, False, False
        else:
            return False, False, False


def normalLineAt_t_toInnerSeg_intersects_withOuter(
        inner_t, inner_seg, outerPath, center, *debug):
    """ Faster version of normalLineAtT_toInner_intersects_withOuter

    Note: This fcn is meant to be used by the
    `IncompleteRing.findTransects2endpointsFromInnerPath_normal` method

    Returns:
        (normal_Line, outer_seg, outer_t) where normal_Line is the Line
            object normal to `inner_path` beginning at
            `inner_path.point(innerT)` and ending at `outer_seg.point(outer_t)`
            or, in the case that the normal line never intersects `outer_path`,
            returns (False, False, False).
    """
    assert 0<= inner_t <= 1
    innerPt = inner_seg.point(inner_t)
    z_deriv = segDerivative(inner_seg,inner_t)
    if z_deriv == 0:
        if inner_t>0.5:
            dt = -0.001
        else:
            dt = 0.001
        k=0
        while z_deriv == 0 and k < 100:
            k+=1
            offset_t = inner_t+k*dt
            if offset_t>1 or offset_t<0:
                raise Exception("offset_t = %s"%offset_t)
            # non-differentiable point encountered
            z_deriv = segDerivative(inner_seg, offset_t)

    # note this should always be outward normal, all curves made CCW earlier
    n_vec = z_deriv.imag - 1j*z_deriv.real
    n_vec = n_vec/abs(n_vec)  # normalize

    # find the max length to the outer
    long_mag = 1.1*abs(extremePointInPath(innerPt, outerPath, 1)[0])
    long_norm_line = Line(innerPt, innerPt+long_mag*n_vec)
    long_norm_path = Path(long_norm_line)
    intersec = sortby(pathXpathIntersections(long_norm_path, outerPath), 2)
    if intersec:
        # this tests if the normal line intersect with the inner (other than
        # at its startpoint) before hitting the outer
        intersec_withInner = sortby(
            pathXpathIntersections(Path(long_norm_line), Path(inner_seg)), 2)

        if len(intersec_withInner) > 1 and intersec_withInner[1][2] < intersec[0][2]:
            if debug == ('debug',):
                return long_norm_line, False, False
            else:
                return False, False, False

        # this block tests if the normal line goes through the center (roughly
        # speaking) and finds a false intersection with outer on the other
        # side of the cross-section
        if not isApproxClosedPath(Path(inner_seg)):
            innerClosingLine = Line(inner_seg.start, inner_seg.end)
            closingLine_self_intersections = \
                pathXpathIntersections(Path(inner_seg), Path(innerClosingLine))
            skip = False
            for inters in closingLine_self_intersections:
                pt = inters[0].point(inters[2])
                if (not isNear(pt, inner_seg.start) and
                    not isNear(pt, inner_seg.end)):
                    skip = True
            if not skip:
                intersec_withClosingLine = sortby(
                    pathXpathIntersections(Path(long_norm_line),
                                           Path(innerClosingLine)), 2)
                if (intersec_withClosingLine and
                    intersec_withClosingLine[0][2] < intersec[0][2] and not
                        isNear(innerPt,
                               intersec_withClosingLine[0][1].point(
                                   intersec_withClosingLine[0][3]))):

                    if debug == ('debug',):
                        return long_norm_line, False, False
                    else:
                        return False, False, False

        # this tests if the normal line goes through the center
        # (roughly speaking) and finds a false intersection with outer
        # on the other side of the cross-section
        if not isApproxClosedPath(outerPath):
            outerClosingLine = Line(outerPath.point(0), outerPath.point(1))
            closingLine_self_intersections = \
                pathXpathIntersections(Path(inner_seg), Path(outerClosingLine))
            skip = False
            for inters in closingLine_self_intersections:
                pt = inters[0].point(inters[2])
                if (not isNear(pt, outerPath[0].start) and
                    not isNear(pt, outerPath[-1].end)):
                    skip = True
            if not skip:
                intersec_withClosingLine_out = \
                    sortby(pathXpathIntersections(Path(long_norm_line),
                                                  Path(outerClosingLine)), 2)
                if (len(intersec_withClosingLine_out) > 0 and
                        intersec_withClosingLine_out[0][2] < intersec[0][2]):
                    if debug == ('debug',):
                        return long_norm_line, False, False
                    else:
                        return False, False, False
        (seg_nl, seg_out, t_nl, t_out) = intersec[0]
        nlin = Line(innerPt, seg_out.point(t_out))
        return nlin, seg_out, t_out
    else:
        if debug:
            return long_norm_line, False, False
        else:
            return False, False, False


def transect_from_angle(angle, startPoint, outerPath, *debug):
    # note: angle should be in [0,1)
    assert 0 <= angle < 1
    innerPt = startPoint
    normal = cos(2*pi*angle) + 1j*sin(2*pi*angle)
    normal = normal/abs(normal)

    # find the max length to the outer
    long_mag = 1.1*abs( extremePointInPath(innerPt, outerPath, 1)[0] )
    long_norm_line = Line(innerPt, innerPt + long_mag*normal)

    # Note: if path is not convex, there could be multiple intersections
    intersec = pathXpathIntersections(Path(long_norm_line), outerPath)
    if intersec:
        intersec.sort(key=itemgetter(2))
        seg_nl, seg_out, t_nl, t_out = intersec[0]
        nlin = Line(innerPt, seg_out.point(t_out))
        return nlin, seg_out, t_out
    else:
        if debug == ('debug',):
            return long_norm_line, False, False
        else:
            return False, False, False


def displaySVGPaths_numbered(pathList,savefile,*colors):
    """creates and saves an svf file displaying the input paths"""
    import svgwrite
    dwg = svgwrite.Drawing(savefile)

    # add white background
    dwg.add(dwg.rect(insert=(0, 0),
                     size=('100%', '100%'),
                     rx=None,
                     ry=None,
                     fill='white'))

    dc = 100/len(pathList)
    for i,p in enumerate(pathList):
        if isinstance(p, Path):
            ps = path2str(p)
        elif isinstance(p, Line) or isinstance(p, CubicBezier):
            ps = path2str(Path(p))
        else:
            ps = p
        if colors != tuple():
            dwg.add(dwg.path(ps, stroke=colors[0][i], fill='none'))
            paragraph = dwg.add(dwg.g(font_size=14))
            coords = (p[0].start.real,p[0].start.imag)
            paragraph.add(dwg.text(str(i), coords))
        else:
            dwg.add(dwg.path(ps,
                             stroke=svgwrite.rgb(0+dc, 0+dc, 16, '%'),
                             fill='none'))
    dwg.save()


def displaySVGPaths(pathList, *colors):
    """creates and saves an svg file displaying the input paths"""
    show_closed_discont=True
    import svgwrite
    from svgpathtools import Path, Line, CubicBezier

    dwg = svgwrite.Drawing('temporary_displaySVGPaths.svg')

    # add white background
    dwg.add(dwg.rect(insert=(0, 0),
                     size=('100%', '100%'),
                     rx=None, ry=None,
                     fill='white'))

    dc = 100/len(pathList)
    for i,p in enumerate(pathList):
        if isinstance(p, Path):
            startpt = p[0].start
            ps = path2str(p)
        elif isinstance(p, Line) or isinstance(p,CubicBezier):
            startpt = p.start
            ps = path2str(p)
        else:
            startpt = parse_path(p)[0].start
            ps = p
        if colors != tuple():
            dwg.add(dwg.path(ps, stroke=colors[0][i], fill='none'))
        else:
            dwg.add(dwg.path(ps,
                             stroke=svgwrite.rgb(0+dc, 0+dc, 16, '%'),
                             fill='none'))
        if show_closed_discont and isApproxClosedPath(p):
            startpt = (startpt.real, startpt.imag)
            dwg.add(dwg.circle(startpt, 1, stroke='gray', fill='gray'))
    dwg.save()


def displaySVGPaths_transects_old(ringList, data_transects,
                                  transect_angles, filename):
    """creates and saves an svf file displaying the input paths"""
    import svgwrite
    transectPaths = []
    for tran_index in range(len(data_transects)):
        tran_path = Path()
        for seg_index in range(len(data_transects[tran_index])-1):
            start_pt = data_transects[tran_index][seg_index]
            end_pt = data_transects[tran_index][seg_index+1]
            tran_path.append(Line(start_pt,end_pt))
        transectPaths.append(tran_path)

    ringPaths = [r.path for r in ringList]
    ringColors = [r.color for r in ringList]
    pathList = ringPaths+transectPaths
    colors = ringColors + ['black']*len(transectPaths)

    # flatten data_transects
    transect_nodes = [item for sublist in data_transects for item in sublist]

    transect_nodes = [(z.real, z.imag) for z in transect_nodes]
    center = ringList[0].center
    center = (center.real, center.imag)

    dwg = svgwrite.Drawing(filename + '_transects.svg',
                           size=('2000px', '2000px'),
                           viewBox="0 0 2000 2000")

    # add white background
    dwg.add(dwg.rect(insert=(0, 0),
                     size=('100%', '100%'),
                     rx=None,
                     ry=None,
                     fill='white'))

    for i, p in enumerate(pathList):
        if isinstance(p, Path):
            ps = path2str(p)
        elif isinstance(p, Line) or isinstance(p, CubicBezier):
            ps = path2str(Path(p))
        else:
            ps = p
        dwg.add(dwg.path(ps, stroke=colors[i], fill='none'))

    # add a purple dot whenever a transect crosses a ring
    for pt in transect_nodes:
        dwg.add(dwg.circle(pt,1, stroke='purple', fill='purple'))

    # add a blue dot at the core/center of the sample
    dwg.add(dwg.circle(center,2, stroke='blue', fill='blue'))

    # add text giving angle in radians/2pi (so in [0,1])
    # at the end of each transect
    for k, theta in enumerate(transect_angles):
        try:
            if len(data_transects[k]) > 1:
                paragraph = dwg.add(dwg.g(font_size=14))
                n_vec = data_transects[k][-1]-data_transects[k][-2]
                n_vec = n_vec/abs(n_vec)
                text_coords = 10*n_vec + data_transects[k][-1]
                text_coords = (text_coords.real, text_coords.imag)
                paragraph.add(dwg.text('%.3f' % theta, text_coords))
            else:
                print('Skipping degenerate transect at angle %s' % theta)
        except:
            print('Skipping problemsome transect at angle %s' % theta)
    dwg.save()


def displaySVGPaths_transects(ring_list, data_transects, transect_angles,
                              skipped_angle_indices, fn=None):
    if not fn:
        filename = opt.output_directory + ring_list[0].svgname
    else:
        filename = fn

    transectPaths = []
    for tran_index in range(len(data_transects)):
        tran_path = Path()
        for seg_index in range(len(data_transects[tran_index]) - 1):
            start_pt = data_transects[tran_index][seg_index]
            end_pt = data_transects[tran_index][seg_index + 1]
            tran_path.append(Line(start_pt,end_pt))
        transectPaths.append(tran_path)

    ringPaths = [r.path for r in ring_list]
    ringColors = [r.color for r in ring_list]
    pathList = ringPaths + transectPaths
    colors = ringColors + ['black']*len(transectPaths)

    # flatten data_transects
    transect_nodes = [item for sublist in data_transects for item in sublist]

    nodes = transect_nodes + [ring_list[0].center]
    node_colors = ['purple']*len(transect_nodes) + ['blue']
    text = ['%.3f' % theta for idx, theta in enumerate(transect_angles)
            if idx not in skipped_angle_indices]
    text += ['skipped %.3f' % transect_angles[idx]
             for idx in skipped_angle_indices]
    text_path = []
    for tr in data_transects:
        end = tr[-1]
        last_seg = Line(tr[-2], tr[-1])
        u = last_seg.unit_tangent(1)
        text_path.append(Path(Line(end + 10*u, end + 100*u)))

    # handle skipped transects
    bdry_ring = max(ring_list, key=lambda ring: ring.maxR)
    bdry_length = bdry_ring.path.length()
    for idx in skipped_angle_indices:
        s = bdry_length * transect_angles[idx]
        T = inv_arclength(bdry_ring.path,  s)
        u = bdry_ring.path.normal(T)
        end = bdry_ring.path.point(T)
        text_path.append(Line(end + 10*u, end + 100*u))

    wsvg(pathList, colors, nodes=nodes, node_colors=node_colors, text=text,
         text_path=text_path, filename=filename+'_transects.svg')


def displaySVGPaths_named(pathList, name, *colors):
    """"DEPRECATED. Creates and saves an svg file displaying the input paths"""
    import svgwrite
    filename = name+'.svg'
    dwg = svgwrite.Drawing(filename,
                           size=('2000px', '2000px'),
                           viewBox="0 0 2000 2000")

    # add white background
    dwg.add(dwg.rect(insert=(0, 0),
                     size=('100%', '100%'),
                     rx=None,
                     ry=None,
                     fill='white'))

    dc = 100 / len(pathList)
    for i, p in enumerate(pathList):
        if isinstance(p, Path):
            ps = path2str(p)
        elif isinstance(p, Line) or isinstance(p, CubicBezier):
            ps = path2str(Path(p))
        else:
            ps = p
        if colors != tuple():
            dwg.add(dwg.path(ps, stroke=colors[0][i], fill='none'))
        else:
            dwg.add(dwg.path(ps,
                             stroke=svgwrite.rgb(0+dc, 0+dc, 16, '%'),
                             fill='none'))
    dwg.save()


def plotUnraveledRings(ring_list,center):
    import matplotlib.pyplot as plt
    import operator
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('theta based on center')
    ax.set_ylabel('radius from center')
    ax.set_title('Straightened Ring Plot')
    for ring in ring_list:
        N = 50  # determines fineness of plot
        x = lambda t: phase(ring.path.point(t)- center)
        y = lambda t: abs(ring.path.point(t)- center)
        tvals = [float(i)/(N-1) for i in range(N)]
        pts = [(x(t),y(t)) for t in tvals]
        pts2 = []
        for i in range(1,len(pts)):
            if abs(pts[i][0] - pts[i-1][0]) > 3:
                pts2 = pts[i:len(pts)].sorted(key=operator.itemgetter(0))
                pts = pts[0:i].sorted(key=operator.itemgetter(0))
                break
        xpts = [pt[0] for pt in pts]
        ypts = [pt[1] for pt in pts]
        ax.plot(xpts, ypts, 'b')
        if pts2 != []:
            xpts = [pt[0] for pt in pts2]
            ypts = [pt[1] for pt in pts2]
            ax.plot(xpts, ypts, 'r')
    fig.show()


def display2rings4user(green_index, red_index, ring_list, mode=None):
    from options4rings import try_to_open_svgs_in_browser, colordict
    filename = 'temporary_4manualSorting.svg'
    save_location = os_path.join(getcwd(),filename)
    center = ring_list[0].center
    boundary_ring = max(ring_list,key=lambda r: r.maxR)
    green = '#00FF00'
    red = '#FF0000'
    bdry_col = colordict['boundary']

    if mode is None or mode == 'b':
        disp_paths = [ring_list[green_index].path, ring_list[red_index].path]
        disp_path_colors = [green, red]
        disp_paths += [ring_list[index].path for index in range(len(ring_list))
                       if index not in [green_index, red_index]]
        disp_path_colors += ['black']*(len(ring_list)-2)
    elif mode == 'g':
        # just display green ring (and boundary and center)
        if boundary_ring == ring_list[green_index]:
            disp_paths = [boundary_ring.path]
            disp_path_colors = [green]
        else:
            disp_paths = [ring_list[green_index].path, boundary_ring.path]
            disp_path_colors = [green,'black']
    elif mode == 'r':
        # just display green ring (and boundary and center)
        if boundary_ring == ring_list[red_index]:
            disp_paths = [boundary_ring.path]
            disp_path_colors = [red]
        else:
            disp_paths = [ring_list[red_index].path, boundary_ring.path]
            disp_path_colors = [red, 'black']
    elif mode == 'b+':
        if boundary_ring == ring_list[green_index] or \
                boundary_ring == ring_list[red_index]:
            disp_paths = [ring_list[green_index].path,
                          ring_list[red_index].path]
            disp_path_colors = [green, red]
        else:
            disp_paths = [ring_list[green_index].path,
                          ring_list[red_index].path,
                          boundary_ring.path]
            disp_path_colors = [green, red, bdry_col]
    elif mode == 'rb':
        assert (boundary_ring != ring_list[green_index] and
                boundary_ring != ring_list[red_index])
        disp_paths = [ring_list[green_index].path,
                      ring_list[red_index].path,
                      boundary_ring.path]
        disp_path_colors = [ring_list[green_index].color,
                            ring_list[red_index].color,
                            bdry_col]
    else:
        Exception("There is no such setting, 'mode=%s'." % mode)

    if mode == 'db':
        disvg(disp_paths + [Line(center-1, center+1)],
              disp_path_colors + [colordict['center']],
              filename=filename, openinbrowser=try_to_open_svgs_in_browser)
    else:
        disvg(disp_paths, disp_path_colors, nodes=[center],
              node_colors=[colordict['center']], filename=filename,
              openinbrowser=try_to_open_svgs_in_browser)
    print('SVG displaying rings at question saved (temporarily) as:\n'
          + save_location)


def closestRing(pt, ring_list):
    """ Finds closest ring in `ring_list` to point, `pt`.

    Returns:
        (|ring.seg.point(t)-pt|, t, seg, ring, i) where t minimizes the
        distance between pt and curve ring.point(t) for 0<=t<=1 and
        ring the closest ring to pt in ring_list; i is the index of ring
        in ring_list
    """
    result_list = []
    for (i, ring) in enumerate(ring_list):
        ring = ring_list[i]
        result_list.append(closestPointInPath(pt,ring.path)+(ring,i))
    return min(result_list,key=itemgetter(0))


class Theta_Tstar(object):
    """made to sort T-values so that they proceed backwards
    (CCW assuming CW path parameterization) from Tstar
    creates an object with a method distfcn that returns the
    theta-distance that (traveling backwards) input is from Tstar.
    This can be used as a key fcn to sort."""
    def __init__(self, Tstar):
        self.Tstar = Tstar

    def distfcn(self, T):
        if T < self.Tstar:
            return self.Tstar - T
        else:
            return 1 - T + self.Tstar


def remove_degenerate_segments(path):
    """This function removes segments that start and end at the same point"""
    new_path = Path()
    for seg in path:
        if seg.start != seg.end:
            new_path.append(seg)
    return new_path


def remove_degenerate_segments_helper(func):
   def func_wrapper(*args, **kwargs):
       return remove_degenerate_segments(func(*args, **kwargs))
   return func_wrapper


def dis(paths, colors=None, nodes=None, node_colors=None, node_radii=None,
        lines=None, line_colors=None,
        filename=os_path.join(getcwd(), 'temporary_displaySVGPaths.svg'),
        openInBrowser=True, stroke_width=opt.stroke_width_default,
        margin_size=0.1):
    """This is the same as disvg, but with openInBrowser=True by default"""
    if lines and paths:
        stroke_widths = [stroke_width] * len(paths + lines)
    elif paths:
        stroke_widths = [stroke_width] * len(paths)
    elif lines:
        stroke_widths = [stroke_width] * len(lines)
    else:
        stroke_widths = None
    disvg(paths + lines, colors + line_colors,
          nodes=nodes, node_colors=node_colors, node_radii=node_radii,
          filename=filename, openinbrowser=openInBrowser,
          stroke_widths=stroke_widths, margin_size=margin_size)
