# This is a tool kit which adds functionality to the python module svg.path
# The functions present here are only designed to deal with Line objects, CubicBezier objects, and Path objects composed of Line and CubicBezier objects
# External Dependencies:
from svgpathtools import Path, Line, CubicBezier, parse_path, polyroots
from numpy import array, dot, roots, isclose, matrix, inf
from scipy.integrate import quad
from operator import itemgetter
from math import sqrt
from os import getcwd, path as os_path, listdir, unlink, makedirs
from xml.dom import minidom

# Parameters:
# tol_isApproxClosedPath_tol = 0 #whether open ended path should be considered closed if it is almost closed
# stroke_width_default = 1 #default stroke-width for output SVGs
# tol_intersections = 10**(-8)
# try_to_open_svgs_in_browser = False
# openFileInBrowser = ??? paste in fcn from andysmod
from options4rings import stroke_width_default, tol_intersections, \
    try_to_open_svgs_in_browser, rings_may_contain_unremoved_kinks
import options4rings as opt
from andysmod import open_in_browser, memoize

check_for_kinksTrue = rings_may_contain_unremoved_kinks  # set to true for github version


###Misc###########################################################################
# def isNear(p,q): #takes two complex numbers, returns boolean
#     return abs(p-q) < tol_isNear
def z2xy(z):
    return z.real, z.imag


def isClosedPathStr(
        pathStr):  # takes in SVF 'd=' path string and outputs true if closed (according to SVG)
    tmp = pathStr.rstrip()
    return tmp[len(tmp) - 1] == 'z'


def isClosed(path_or_seg):
    if isinstance(path_or_seg, Path):
        return path_or_seg[-1].end == path_or_seg[0].start
    elif isinstance(path_or_seg, CubicBezier) or isinstance(path_or_seg, Line):
        return path_or_seg.end == path_or_seg.start
    else:
        raise Exception("First argument must be Path or CubicBezier or Line.")


def cubicCurve(P,
               t):  # Evaluated the cubic Bezier curve (given by control points P) at t
    return P[0] * (1 - t) ** 3 + 3 * P[1] * t * (1 - t) ** 2 + 3 * P[2] * (
    1 - t) * t ** 2 + P[3] * t ** 3


def concatPaths(list_of_paths):
    return Path(*[seg for path in list_of_paths for seg in path])


def bezier2standard(cub):
    '''
    Input: CubicBezier object or tuple (P0,P1,P2,P3)
    Output: (p0,p1,p2,p3) s.t. p3*t**3 + p2*t**2 + p1*t + p0 =
    P0*(1-t)**3 + 3*P1*(1-t)**2*t + 3*P2*(1-t)*t**2 + P3*t**3
    '''
    if isinstance(cub, CubicBezier):
        P = cub.bpoints()
    #    A = matrix([[1,3,3,0],
    #                [-3,-6,-6,0],
    #                [3,3,3,0],
    #                [-1,0,0,1]])
    #    res = A*matrix(P).T
    #    return [complex(z[0]) for z in array(res)]
    else:
        P = cub
    return P[0], -3 * P[0] + 3 * P[1], 3 * P[0] - 6 * P[1] + 3 * P[2], -P[
        0] + 3 * P[1] - 3 * P[2] + P[3]


def standard2bezier(coeffs):
    '''
    Input: iterable coeffs = (p0,p1,p2,p3)
    Output: [P0,P1,P2,P3] s.t. p3*t**3 + p2*t**2 + p1*t + p0 =
    P0*(1-t)**3 + 3*P1*(1-t)**2*t + 3*P2*(1-t)*t**2 + P3*t**3
    '''
    A = matrix([[1, 0, 0, 0],
                [-3, 3, 0, 0],
                [3, -6, 3, 0],
                [-1, 3, -3, 1]])
    res = A ** (-1) * matrix(coeffs).T
    return [complex(z[0]) for z in array(res)]


def stcubic_eval(coeffs, t):
    '''
    Input: iterable coeffs = (p0,p1,p2,p3)
    Output: p3*t**3 + p2*t**2 + p1*t + p0
    '''
    p0, p1, p2, p3 = coeffs
    return p3 * t ** 3 + p2 * t ** 2 + p1 * t + p0


def segt2PathT(path, seg, t):  # formerly segt2PathT
    # finds T s.t path(T) = seg(t), for any seg in path and t in [0,1]
    #    path._calc_lengths()
    #    # Find which segment the point we search for is located on:
    #    segment_start = 0
    #    for index, segment in enumerate(path._segments):
    #        segment_end = segment_start + path._lengths[index]
    #        if segment == seg:# then this is the segment! How far in on the segment is the point?
    #            T = (segment_end - segment_start)*t + segment_start
    #            break
    #        segment_start = segment_end
    #    else:
    #        raise Exception("seg was not found to be a segment in path.")
    #    return T
    return path.t2T(seg, t)


def reverseSeg(
        seg):  # reverses the orientation of a Line or CubicBezier segment
    #    if isinstance(seg,CubicBezier):
    #        new_cub = CubicBezier(seg.end,seg.control2,seg.control1,seg.start)
    #        if seg._length:
    #            new_cub._length = seg.length()
    #            new_cub._oldPoints = (new_cub.start,new_cub.control1,new_cub.control2,new_cub.end)
    #        return new_cub
    #    elif isinstance(seg,Line):
    #        return Line(seg.end,seg.start)
    #    else:
    #        raise Exception("Input argument must be a Line object nor a CubicBezier object.")
    return seg.reversed()


def reversePath(
        path):  # reverses the orientation of a path object composed of lines and cubicBeziers segments
    newpath = []
    for seg in path:
        newpath.append(reverseSeg(seg))
    newpath.reverse()
    return Path(*newpath)


def ptInPath2tseg(pt, path, tol=0.1):  # formerly pt2tvalinPath
    # returns (t,seg) where seg in path, 0<=t<=1, seg.point(t) = pt and d = |seg.point(t)-pt|
    (d, t, seg) = closestPointInPath(pt, path)
    if d < tol:
        return (t, seg)
    else:
        raise Exception("pt not in path by tolerance check.  d = %s" % d)


def pathT2tseg(path, T):  # formerly pathT2seg
    # finds which seg the path T value lies on
    #    (t,seg) = ptInPath2tseg(path.point(T),path)
    #    return (t,seg)
    seg_idx, t = path.T2t(T)
    return t, path[seg_idx]


def seg_index(path, seg):
    try:
        return path.index(seg)
    except ValueError:
        raise Exception("seg was not found to be a segment in path.")


####Conversion###########################################################################

def linePts2pathStr(p0, p1):  # formerly pts2lineStr
    # returns the path string for the line connecting these two points
    return 'M' + str(p0.real) + ' ' + str(p0.imag) + 'L' + str(
        p1.real) + ' ' + str(p1.imag)


def cubicPoints2String(P):
    return "M %s,%s C %s,%s %s,%s %s,%s" % (
    P[0].real, P[0].imag, P[1].real, P[1].imag, P[2].real, P[2].imag,
    P[3].real, P[3].imag)


def cubicPoints2RelString(P):
    rP = [p - P[0] for p in P]
    return "M %s,%s c %s,%s %s,%s %s,%s" % (
    P[0].real, P[0].imag, rP[1].real, rP[1].imag, rP[2].real, rP[2].imag,
    rP[3].real, rP[3].imag)


def cubPoints(cubicbezier):
    return cubicbezier.start, cubicbezier.control1, cubicbezier.control2, cubicbezier.end


def polylineStr2pathStr(polylineStr):
    # converts a polyline string (set of points) to be a path string
    # returns the path string
    points = polylineStr.replace(', ', ',')
    points = points.replace(' ,', ',')
    points = points.split()
    try:
        if points[0] == points[-1]:
            closed = True
        else:
            closed = False
    except:
        print points
        raise Exception()
    d = 'M' + points.pop(0).replace(',', ' ')
    for p in points:
        d += 'L' + p.replace(',', ' ')
    if closed:
        d += 'z'
    return d


###Geometric###########################################################################

def extremePointInPath(pt, path, min_or_max):  # 0=min, 1=max
    # returns (|path.seg.point(t)-pt|,t,seg) where t minimizes/maximizes the distance between pt and curve path.point(t) #returns abs of area enclosed by (closed) curve path.point(t) for t in [0,1]
    if min_or_max not in {0, 1}:
        raise Exception(
            "min_or_max must be 0 or 1.\n 0 if minimizer is desired, 1 if maximizer is desired.")
    p, q = pt.real, pt.imag
    current_extremizer = None  ###should be based on native res
    for seg in path:
        if isinstance(seg, CubicBezier):
            (a_0, a_1, a_2, a_3) = (
            seg.start.real, seg.control1.real, seg.control2.real, seg.end.real)
            (b_0, b_1, b_2, b_3) = (
            seg.start.imag, seg.control1.imag, seg.control2.imag, seg.end.imag)
            dist = lambda t: ((a_0 * (t - 1) ** 3 - 3 * a_1 * (
            t - 1) ** 2 * t + 3 * a_2 * (
                               t - 1) * t ** 2 - a_3 * t ** 3 + p) ** 2 + (
                              b_0 * (t - 1) ** 3 - 3 * b_1 * (
                              t - 1) ** 2 * t + 3 * b_2 * (
                              t - 1) * t ** 2 - b_3 * t ** 3 + q) ** 2) ** (
                             0.5)  # gives the distance from pt to path.point(t)
            polycoeffs_dist_deriv = [
                6 * (a_0 - 3 * a_1 + 3 * a_2 - a_3) ** 2 + 6 * (
                b_0 - 3 * b_1 + 3 * b_2 - b_3) ** 2,
                -30 * (a_0 - 2 * a_1 + a_2) * (
                a_0 - 3 * a_1 + 3 * a_2 - a_3) - 30 * (b_0 - 2 * b_1 + b_2) * (
                b_0 - 3 * b_1 + 3 * b_2 - b_3),
                36 * (a_0 - 2 * a_1 + a_2) ** 2 + 24 * (a_0 - a_1) * (
                a_0 - 3 * a_1 + 3 * a_2 - a_3) + 36 * (
                b_0 - 2 * b_1 + b_2) ** 2 + 24 * (b_0 - b_1) * (
                b_0 - 3 * b_1 + 3 * b_2 - b_3),
                -54 * (a_0 - a_1) * (a_0 - 2 * a_1 + a_2) - 6 * (
                a_0 - 3 * a_1 + 3 * a_2 - a_3) * (a_0 - p) - 54 * (
                b_0 - b_1) * (b_0 - 2 * b_1 + b_2) - 6 * (
                b_0 - 3 * b_1 + 3 * b_2 - b_3) * (b_0 - q),
                18 * (a_0 - a_1) ** 2 + 12 * (a_0 - 2 * a_1 + a_2) * (
                a_0 - p) + 18 * (b_0 - b_1) ** 2 + 12 * (
                b_0 - 2 * b_1 + b_2) * (b_0 - q),
                -6 * (a_0 - a_1) * (a_0 - p) - 6 * (b_0 - b_1) * (
                b_0 - q)]  # coefficients for polynomial (d/dt)dist = (d/dt)|path.point(t)-pt|)
            possible_extremizers = list(roots(polycoeffs_dist_deriv)) + [0, 1]
            for t in possible_extremizers:
                if t.real == t and 0 <= t <= 1:
                    t = t.real
                    try:
                        if dist(t) * (-1) ** min_or_max < current_extremizer[
                            0] * (-1) ** min_or_max:
                            current_extremizer = (dist(t), t, seg)
                    except:
                        if current_extremizer == None:
                            current_extremizer = (dist(t), t, seg)
                        else:
                            raise Exception("Something's wrong")
        elif isinstance(seg, Line):
            (a_0, a_1) = (seg.start.real, seg.end.real)
            (b_0, b_1) = (seg.start.imag, seg.end.imag)
            polycoeffs_dist_deriv = [
                2 * (a_0 - a_1) ** 2 + 2 * (b_0 - b_1) ** 2,
                -2 * (a_0 - a_1) * (a_0 - p) - 2 * (b_0 - b_1) * (b_0 - q)]
            dist = lambda t: ((a_0 * (t - 1) - a_1 * t + p) ** 2 + (
            b_0 * (t - 1) - b_1 * t + q) ** 2) ** (0.5)
            possible_extremizers = list(roots(polycoeffs_dist_deriv)) + [0, 1]
            for t in possible_extremizers:
                if t.real == t and 0 <= t <= 1:
                    t = t.real
                    try:
                        if dist(t) * (-1) ** min_or_max < current_extremizer[
                            0] * (-1) ** min_or_max:
                            current_extremizer = (dist(t), t, seg)
                    except:
                        if current_extremizer == None:
                            current_extremizer = (dist(t), t, seg)
                        else:
                            raise Exception("Something's wrong")
        else:
            raise Exception(
                'Path segment is neither Line object nor CubicBezier object.')
    return current_extremizer


def closestPointInPath(pt, path):
    # returns (|path.seg.point(t)-pt|,t,seg) where t minimizes the distance between pt and curve path.point(t) for t in [0,1]
    return extremePointInPath(pt, path, 0)


def closestPath(pt, path_list):
    # returns (|path.seg.point(t)-pt|,t,seg,path) where t minimizes the distance between pt and curve path.point(t) for 0<=t<=1 and path the closest path to pt in path_list
    result_list = []
    for path in path_list:
        (d, t, seg) = closestPointInPath(pt, path)
        result_list.append(closestPointInPath(pt, path) + (path,))
    return min(result_list, key=itemgetter(0))


def minRadius(origin, path):
    # returns |path.seg.point(t)-pt| where t minimizes the distance between origin and curve path.point(t) for 0<=t<=1
    return extremePointInPath(origin, path, 0)[0]


def maxRadius(origin, path):
    # returns |path.seg.point(t)-pt| where t maximizes the distance between pt and curve path.point(t) for 0<=t<=1
    return extremePointInPath(origin, path, 1)[0]


def segDerivative(seg, t):
    '''
    This returns the derivative of seg at t
    Note: This will be a positive scalar multiple of the derivative of the Path
    seg is part of (at the corresponding T)
    '''
    if isinstance(seg, CubicBezier):
        P = (seg.start, seg.control1, seg.control2, seg.end)
        return 3 * (P[1] - P[0]) * (1 - t) ** 2 + 6 * (P[2] - P[1]) * (
        1 - t) * t + 3 * (P[3] - P[2]) * t ** 2
    elif isinstance(seg, Line):
        P = (seg.start, seg.end)
        return P[1] - P[0]
    else:
        raise Exception("First argument must be a Line or a CubicBezier.")


def segUnitTangent(seg, t):
    assert 0 <= t <= 1
    dseg = segDerivative(seg, t)
    try:
        tangent = (dseg + 0.0) / abs(dseg)
    except ZeroDivisionError:
        from andysmod import limit
        def tangentfunc(tval):
            ds = segDerivative(seg, tval)
            return ds / abs(ds)

        if isclose(t, 0):
            side = 1
            delta0 = .01
        elif isclose(t, 1):
            side = -1
            delta0 = .01
        else:
            side = 0
            delta0 = min(0.01, 1 - t, t)
        tangent = limit(tangentfunc, t, side=side, delta0=delta0)
    return tangent


def unitTangent(path_or_seg, T_or_t):
    if isinstance(path_or_seg, Path):
        t, seg = pathT2tseg(path_or_seg, T_or_t)
        return segUnitTangent(seg, t)
    else:
        return segUnitTangent(path_or_seg, T_or_t)


def normal(path_or_seg, T_or_t):
    if isinstance(path_or_seg, Path):
        t, seg = pathT2tseg(path_or_seg, T_or_t)
    else:
        seg = path_or_seg
        t = T_or_t
    n = -1j * segDerivative(seg, t)
    return n / abs(n)


def segCurvature(seg, t,
                 use_limit_for_zero_divs=True):  # This returns the curvature of the segment at t
    #    if isinstance(seg,CubicBezier):
    #        P = (seg.start,seg.control1,seg.control2,seg.end)
    #        dz = 3*(P[1]-P[0])*(1-t)**2 + 6*(P[2]-P[1])*(1-t)*t + 3*(P[3]-P[2])*t**2
    #        ddz = 6*((1-t)*(P[2]-2*P[1]+P[0])+t*(P[3]-2*P[2]+P[1]))
    #        dx,dy = dz.real,dz.imag
    #        ddx,ddy = ddz.real,ddz.imag
    #
    #        try:
    #            kappa = abs(dx*ddy-dy*ddx)/(dx*dx+dy*dy)**(1.5)
    #        except ZeroDivisionError:
    #            if use_limit_for_zero_divs:
    #                from andysmod import limit
    #                tempfunc = lambda t: segCurvature(seg,t,use_limit_for_zero_divs = False)
    #                kappa = limit(tempfunc,t)
    #            else:
    #                raise
    #        return kappa
    #    elif isinstance(seg,Line):
    #        return 0
    #    else:
    #        raise Exception("First argument must be a Line or a CubicBezier.")
    return seg.curvature(t)


def curvature(path_or_seg, T_or_t, check_for_kinks=check_for_kinksTrue):
    '''
    outputs numpy.inf if point is a corner (i.e. is continous but
    non-differentiable.)
    '''
    if check_for_kinks and isinstance(path_or_seg, Path):
        path = path_or_seg
        t, seg = pathT2tseg(path, T_or_t)
        seg_idx = path.index(seg)
        if isclose(t, 0) and (seg_idx != 0 or isClosed(path)):
            if not isclose(normal(seg, 0),
                           normal(path[(seg_idx - 1) % len(path)], 1)):
                return inf
        elif isclose(t, 1) and (seg_idx != len(path) - 1 or isClosed(path)):
            if not isclose(normal(seg, 1),
                           normal(path[(seg_idx + 1) % len(path)], 0)):
                return inf
    if isinstance(path_or_seg, Line):
        return 0
    elif isinstance(path_or_seg, CubicBezier):
        return segCurvature(path_or_seg, T_or_t)
    elif isinstance(path_or_seg, Path):
        (t, seg) = pathT2tseg(path_or_seg, T_or_t)
        return segCurvature(seg, t)
    else:
        raise Exception(
            "First argument must be a Line, CubicBezier, or Path containing only Lines and CubicBeziers.")


def pathSegDerivative(path, T):
    '''This returns the derivative of the segment of path that T lies on, at t... so is a scalar multiple of the derivative of path.point(T) at T'''
    (t, seg) = pathT2tseg(path, T)
    if isinstance(seg, CubicBezier):
        P = (seg.start, seg.control1, seg.control2, seg.end)
        return 3 * (P[1] - P[0]) * (1 - t) ** 2 + 6 * (P[2] - P[1]) * (
        1 - t) * t + 3 * (P[3] - P[2]) * t ** 2
    elif isinstance(seg, Line):
        P = (seg.start, seg.end)
        return P[1] - P[0]


def printPath(path):
    """Prints path in a nice way.  path should be a Path, CubicBezier, Line, or
        list of CubicBezier objects and Line objects."""
    if not isinstance(path, Path):
        if isinstance(path, Line) or isinstance(path, CubicBezier):
            path = Path(path)
        elif all([isinstance(seg, Line) or isinstance(seg, CubicBezier) for seg
                  in path]):
            path = Path(*path)
        else:
            print "This path is not a path... and is neither a Line object nor a CubicBezier object."
            return
    try:
        path[0]
    except IndexError:
        print "This path seems to be empty."
        return

    output_string = ""

    for seg_index_, seg in enumerate(path):
        if seg_index_ != 0:
            output_string += "\n"
        if isinstance(seg, CubicBezier):
            tmp = []
            for z in cubPoints(seg):
                tmp += [z.real, z.imag]
            nicePts = "(%.1f + i%.1f, %.1f + i%.1f, %.1f + i%.1f, %.1f + i%.1f)" % tuple(
                tmp)
            output_string += "[%s] - CubicBezier: " % seg_index_ + nicePts
        elif isinstance(seg, Line):
            nicePts = "(%.1f + i%.1f, %.1f + i%.1f)" % (
            seg.start.real, seg.start.imag, seg.end.real, seg.end.imag,)
            output_string += "[%s] - Line       : " % seg_index_ + nicePts
        else:
            print("+" * 50)
            print(seg)
            raise Exception("This path contains a segment that is neither a Line nor a CubicBezier.")
    if path[0].start == path[-1].end:
        closure = "Closed"
    else:
        closure = "Open  "
    output_string += "\n" + "[*] " + closure + " : |path.point(0) - path.point(1)| = %s" % abs(
        path.point(0) - path.point(1))
    print(output_string)


def areaEnclosed(path, closure_tolerance=opt.tol_isApproxClosedPath):
    """returns absolute value of area enclosed by path note: negative area
    results from CW (as oppossed to CCW) parameterization of path"""
    if not path.end==path.start and abs(path[0].start - path[-1].end) >= closure_tolerance:
        raise Exception('Path encloses no area.')
    area = 0
    area_error = 0
    for seg in path:
        if isinstance(seg, CubicBezier):
            [a_0, a_1, a_2, a_3] = [z.real for z in seg.bpoints()]
            [b_0, b_1, b_2, b_3] = [z.imag for z in seg.bpoints()]
            x = lambda t: a_0 * (1 - t) ** 3 + 3 * a_1 * t * (
                                                             1 - t) ** 2 + 3 * a_2 * (
            1 - t) * t ** 2 + a_3 * t ** 3  # real part of cubic Bezier
            dy = lambda t: 3 * (b_1 - b_0) * (1 - t) ** 2 + 6 * (
            b_2 - b_1) * t * (1 - t) + 3 * (
            b_3 - b_2) * t ** 2  # derivative of imaginary part of cubic Bezier
            integrand = lambda t: x(t) * dy(t)
        elif isinstance(seg, Line):
            x = lambda t: (1 - t) * seg.start.real + t * seg.end.real
            dy = seg.end.imag - seg.start.imag
            integrand = lambda t: x(t) * dy
        else:
            raise Exception(
                'Path segment is neither Line object nor CubicBezier object.')
        result = quad(integrand, 0, 1)
        area += result[0]
        area_error += result[1]
    if area_error > 1e-2:
        from warnings import warn
        warn('\narea may be erroneous (by aprrox %s)\n' % area_error)
    return abs(area)


###Intersections###########################################################################
def pathXpathIntersections(path1, path2, justonemode=False, tol=1e-4):
    """returns list of (seg1, seg2, t1, t2) tuples."""
    return [(inter[0][1], inter[1][1], inter[0][2], inter[1][2]) for inter in
            path1.intersect(path2)]


# def pathXpathIntersections(path1,path2,justonemode=False,tol=10**(-4)):
#     """returns list of (seg1, seg2, t1, t2) tuples describing the intersection points
#     if justonemode=True, then returns just the first intersection found
#     tol is used to check for redundant intersections (see comment above the code block where tol is used)"""
#     if path1==path2:
#         return []
#     intersection_list = []
#     for seg1 in path1:
#         for seg2 in path2:
#             if justonemode and len(intersection_list) > 0:
#                 return intersection_list[0]
#             if isinstance(seg1,CubicBezier) and isinstance(seg2,CubicBezier):
#                 seg1_points = (seg1.start, seg1.control1, seg1.control2, seg1.end)
#                 seg2_points = (seg2.start, seg2.control1, seg2.control2, seg2.end)
#                 for (t1,t2) in cubicXcubicIntersections((seg1_points,seg2_points)):
#                     intersection_list.append((seg1,seg2,t1,t2))
#             elif isinstance(seg1,CubicBezier) and isinstance(seg2,Line):
#                 for (t1,t2) in cubicXlineIntersections(seg1,seg2):
#                     intersection_list.append((seg1,seg2,t1,t2))
#             elif isinstance(seg1,Line) and isinstance(seg2,CubicBezier):
#                 for (t2,t1) in cubicXlineIntersections(seg2,seg1):
#                     intersection_list.append((seg1,seg2,t1,t2))
#             elif isinstance(seg1,Line) and isinstance(seg2,Line):
#                 for (t1,t2) in lineXlineIntersections(seg1,seg2):
#                     intersection_list.append((seg1,seg2,t1,t2))
#             else:
#                 if not (isinstance(seg1,Line) or isinstance(seg1,CubicBezier)):
#                     print("path1 = " + str(path1))
#                     raise  Exception("Segment from path1 is not line or cubic bezier.")
#                 elif not (isinstance(seg2,Line) or isinstance(seg2,CubicBezier)):
#                     print("path2 = " + str(path2))
#                     raise  Exception("Segment from path2 is not line or cubic bezier.")
#                 raise Exception("I dunno what's wrong, but somethin.")
#     if justonemode and len(intersection_list) > 0:
#         return intersection_list[0]
#
#     #Note: If the intersection takes place at a joint (point one seg ends and next begins in path) then intersection_list may contain a reduntant intersection.  This code block checks for and removes said redundancies
#     pts = [seg1.point(t1) for (seg1,seg2,t1,t2) in intersection_list]
#     indices2remove = []
#     for ind1 in range(len(pts)):
#         for ind2 in range(ind1+1,len(pts)):
#             if abs(pts[ind1]-pts[ind2])<tol:
#             #then there's a redundancy. Remove it.
#                 indices2remove.append(ind2)
#     intersection_list = [inter for ind,inter in enumerate(intersection_list) if ind not in indices2remove]
#     return intersection_list


def path_listXpathIntersections(path_list,
                                path):  # returns list of lists of tuples (seg,seg_o,t1,t2)
    return [pathXpathIntersections(path, otherPath) for otherPath in path_list]


def pathXlineIntersections(inline, path):
    intersection_list = []
    for seg in path:
        if isinstance(seg, CubicBezier):
            for (tp, tl) in cubicXlineIntersections(seg, inline):
                intersection_list.append((tl, seg, tp))
        elif isinstance(seg, Line):
            for (tp, tl) in lineXlineIntersections(seg, inline):
                intersection_list.append((tl, seg, tp))
        else:
            raise Exception(
                "seg is not in a Line object or Cubic Bezier object.")
    return intersection_list


def pathlistXlineIntersections(inline, path_list):
    # Takes in list of paths and returns list of tuples (tl,path_index,seg,tp) where line.point(tl) = seg.point(tp) where seg is a segment in path_list[index]
    return [(tl, path_index, seg, tp) for path_index, path in
            enumerate(path_list) for (tl, seg, tp) in
            pathXlineIntersections(inline, path)]


def cubicXlineIntersections(cub, line):
    # the method here is to rotate the line to be the x-axis (and the cubic with it), then the intersection points are just the roots of the rotated cubic.
    if line.start == line.end:
        return []
    lP0, lP1 = array([line.start.real, line.start.imag]), array(
        [line.end.real, line.end.imag])
    cP0, cP1, cP2, cP3 = array([cub.start.real, cub.start.imag]), array(
        [cub.control1.real, cub.control1.imag]), array(
        [cub.control2.real, cub.control2.imag]), array(
        [cub.end.real, cub.end.imag])
    lQ0, lQ1, cQ0, cQ1, cQ2, cQ3 = array([0,
                                          0]), lP1 - lP0, cP0 - lP0, cP1 - lP0, cP2 - lP0, cP3 - lP0  # translates all points so that the line goes through the origin (with P0=0)
    a = lQ1[0];
    b = lQ1[1]
    if a:
        R = array([[1 / a, 0], [-b / a, 1]])  # inverse of [[a,0],[b,1]]
    else:
        R = array([[0, 1 / b], [1, -a / b]])  # inverse of [[a,1],[b,0]]
    lQ1, cQ0, cQ1, cQ2, cQ3 = dot(R, lQ1), dot(R, cQ0), dot(R, cQ1), dot(R,
                                                                         cQ2), dot(
        R,
        cQ3)  # rotates all points so that the line goes along the x-axis the origin (with P0=(0,0) and P1 = (1,0))
    # Now the intersection points are simply the points where the cubic crosses the x-axis (i.e. the roots of y(t))
    b_0, b_1, b_2, b_3 = cQ0[1], cQ1[1], cQ2[1], cQ3[1]
    coeffs_y = [-b_0 + 3 * b_1 - 3 * b_2 + b_3, 3 * b_0 - 6 * b_1 + 3 * b_2,
                -3 * b_0 + 3 * b_1, b_0]
    roots_y = list(roots(coeffs_y))
    #    roots_y = list(cubroots_nr(coeffs_y)) #numpy newton-ralphon tests as slower, so using numpy roots
    a = (
    cQ0[0], cQ1[0], cQ2[0], cQ3[0])  # to find x-values associated with y(t)=0
    intersection_list = []
    for r in set(roots_y):
        if not r.imag and 0 <= r <= 1:
            x_val = cubicCurve(a, r)
            if 0 <= x_val <= 1:
                intersection_list.append((r,
                                          x_val))  # note, after rotation line is parameterized by x(t)=t and y(t)=0
            #    if intersection_list !=[]:
            #        print "lineXcubic intersections found:"
            #        for (t1,t2) in intersection_list:
            #            print "cub(%s) = line(%s) = %s\n"%(t1,t2,cubicCurve((cP0,cP1,cP2,cP3),t1))
    return intersection_list


def lineXlineIntersections(seg1, seg2):
    if seg1.start == seg1.end or seg2.start == seg2.end:
        return []
    p = (seg1.start, seg1.end)
    q = (seg2.start, seg2.end)
    a = [z.real for z in p]
    b = [z.imag for z in p]
    c = [z.real for z in q]
    d = [z.imag for z in q]
    denom = (b[0] - b[1]) * c[0] - (b[0] - b[1]) * c[1] - a[0] * (
    d[0] - d[1]) + a[1] * (d[0] - d[1])
    if denom == 0:
        return []
    t1 = ((b[0] - d[1]) * c[0] - (b[0] - d[0]) * c[1] - a[0] * (
    d[0] - d[1])) / denom
    t2 = -(
    a[1] * (b[0] - d[0]) - a[0] * (b[1] - d[0]) - (b[0] - b[1]) * c[0]) / denom
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return [(t1, t2)]
    return []


def cubicMinMax_x(points):
    '''returns the min and max x coordinates for any point in the cubic'''
    local_extremizers = [0, 1]
    a = [p.real for p in points]
    denom = a[0] - 3 * a[1] + 3 * a[2] - a[3]
    if denom != 0:
        delta = a[1] ** 2 - (a[0] + a[1]) * a[2] + a[2] ** 2 + (a[0] - a[1]) * \
                                                               a[3]
        if delta >= 0:  # otherwise no local extrema
            sqdelta = sqrt(delta)
            tau = a[0] - 2 * a[1] + a[2]
            r1 = (tau + sqdelta) / denom
            r2 = (tau - sqdelta) / denom
            if 0 < r1 < 1:
                local_extremizers.append(r1)
            if 0 < r2 < 1:
                local_extremizers.append(r2)
    else:
        coeffs = list(bezier2standard(a))
        dcoeffs = [k * c for k, c in enumerate(coeffs)]
        dcoeffs.reverse()
        dcoeffs = dcoeffs[0:-1]
        local_extremizers += polyroots(dcoeffs, condition=lambda r: 0 < r < 1,
                                       realroots=True)
    localExtrema = [cubicCurve(a, t) for t in local_extremizers]
    return min(localExtrema), max(localExtrema)


def cubicMinMax_y(points):
    return cubicMinMax_x([-1j * p for p in points])


def intervalIntersectionWidth(a, b, c,
                              d):  # returns width of the intersection of intervals [a,b] and [c,d]  (thinking of these as intervals on the real number line)
    return max(0, min(b, d) - max(a, c))


def cubicBoundingBoxesIntersect(
        cubs):  # INPUT: 2-tuple of cubics (given bu control points) #OUTPUT: boolean
    x1min, x1max = cubicMinMax_x(cubs[0])
    y1min, y1max = cubicMinMax_y(cubs[0])
    x2min, x2max = cubicMinMax_x(cubs[1])
    y2min, y2max = cubicMinMax_y(cubs[1])
    if intervalIntersectionWidth(x1min, x1max, x2min,
                                 x2max) and intervalIntersectionWidth(y1min,
                                                                      y1max,
                                                                      y2min,
                                                                      y2max):
        return True
    else:
        return False


def cubicBoundingBoxArea(cub_points):
    '''
    INPUT: 2-tuple of cubics (given by control points)
    OUTPUT: boolean
    '''
    xmin, xmax = cubicMinMax_x(cub_points)
    ymin, ymax = cubicMinMax_y(cub_points)
    return (xmax - xmin) * (ymax - ymin)


def halveCubic(P):
    return ([P[0], (P[0] + P[1]) / 2, (P[0] + 2 * P[1] + P[2]) / 4,
             (P[0] + 3 * P[1] + 3 * P[2] + P[3]) / 8],
            [(P[0] + 3 * P[1] + 3 * P[2] + P[3]) / 8,
             (P[1] + 2 * P[2] + P[3]) / 4, (P[2] + P[3]) / 2, P[3]])


class Pair(object):
    def __init__(self, cub1, cub2, t1, t2):
        self.cub1 = cub1
        self.cub2 = cub2
        self.t1 = t1  # the t value to get the mid point of this curve from cub1
        self.t2 = t2  # the t value to get the mid point of this curve from cub2


class AppPointSet(list):
    def __init__(self, tol):
        self.tol = tol

    def __contains__(self, pt):
        for x in self:
            if abs(x - pt) < self.tol:
                return True
        return False

    def apsadd(self, pt):
        if pt not in self:
            self.append(pt)


def cubicXcubicIntersections(cubs, Tol_deC=tol_intersections):
    # INPUT: a tuple cubs=([P0,P1,P2,P3], [Q0,Q1,Q2,Q3]) defining the two cubic to check for intersections between.  See cubicCurve fcn for definition of P0,...,P3
    # OUTPUT: a list of tuples (t,s) in [0,1]x[0,1] such that cubicCurve(cubs[0],t) - cubicCurve(cubs[1],s) < Tol_deC
    # Note: This will return exactly one such tuple for each intersection (assuming Tol_deC is small enough)
    maxIts = 100  ##### This should be k = 1-log(Tol_deC/length)/log(2), where length is length of longer cubic
    tol_repeatedIntersections = 2  #####
    pair_list = [Pair(cubs[0], cubs[1], 0.5, 0.5)]
    intersection_list = []
    k = 0
    approx_point_set = AppPointSet(tol_repeatedIntersections)
    while pair_list != []:
        newPairs = []
        delta = 0.5 ** (k + 2)
        for pair in pair_list:
            if cubicBoundingBoxesIntersect((pair.cub1, pair.cub2)):
                if cubicBoundingBoxArea(
                        pair.cub1) < Tol_deC and cubicBoundingBoxArea(
                        pair.cub2) < Tol_deC:
                    point = cubicCurve(cubs[0], pair.t1)
                    if point not in approx_point_set:
                        approx_point_set.append(point)
                        intersection_list.append((pair.t1,
                                                  pair.t2))  # this is the point in the middle of the pair
                    for otherPair in pair_list:
                        if pair.cub1 == otherPair.cub1 or pair.cub2 == otherPair.cub2 or pair.cub1 == otherPair.cub2 or pair.cub2 == otherPair.cub1:
                            pair_list.remove(
                                otherPair)  # this is just an ad-hoc fix to keep it from repeating intersection points
                else:
                    (c11, c12) = halveCubic(pair.cub1)
                    (t11, t12) = (pair.t1 - delta, pair.t1 + delta)
                    (c21, c22) = halveCubic(pair.cub2)
                    (t21, t22) = (pair.t2 - delta, pair.t2 + delta)
                    newPairs += [Pair(c11, c21, t11, t21),
                                 Pair(c11, c22, t11, t22),
                                 Pair(c12, c21, t12, t21),
                                 Pair(c12, c22, t12, t22)]
        pair_list = newPairs
        k += 1
        if k > maxIts:
            raise Exception(
                "cubicXcubicIntersections has reached maximum iterations without terminating... either there's a problem/bug or you can fix by raising the max iterations or lowering Tol_deC")
    return intersection_list


def splitBezier(points, t):
    # returns 2 tuples of control points for the two resulting Bezier curves
    points_left = []
    points_right = []
    (points_left, points_right) = splitBezier_deCasteljau_recursion(
        (points_left, points_right), points, t)
    points_right.reverse()

    ###DEBUG ONLY KLJLKjlkjlkjlkjlkjsdf
    assert points_left[0] == points[0]
    assert points_right[3] == points[3]
    assert points_left[3] == points_right[0]
    ###end of DEBUG ONLY KLJLKjlkjlkjlkjlkjsdf
    return (points_left, points_right)


def splitBezier_deCasteljau_recursion(cub_lr, points, t):
    # Note: This works for not just cubics but any bezier curve
    (cub_left, cub_right) = cub_lr
    if len(points) == 1:
        cub_left.append(points[0])
        cub_right.append(points[0])
    else:
        n = len(points) - 1
        newPoints = [None] * n
        cub_left.append(points[0])
        cub_right.append(points[n])
        for i in range(n):
            newPoints[i] = (1 - t) * points[i] + t * points[i + 1]
        (cub_left, cub_right) = splitBezier_deCasteljau_recursion(
            (cub_left, cub_right), newPoints, t)
    return (cub_left, cub_right)


def trimSeg(seg, t0, t1):
    assert t0 < t1
    if isinstance(seg, CubicBezier):
        if t0 == 0:
            (P0, P1, P2, P3) = splitBezier(cubPoints(seg), t1)[0]
            new_seg = CubicBezier(P0, P1, P2, P3)
        elif t1 == 1:
            (P0, P1, P2, P3) = splitBezier(cubPoints(seg), t0)[1]
            new_seg = CubicBezier(P0, P1, P2, P3)
        else:
            pt1 = seg.point(t1)
            seg_minus_start = trimSeg(seg, t0, 1)
            t1_adj = ptInPath2tseg(pt1, Path(seg_minus_start))[0]
            new_seg = trimSeg(seg_minus_start, 0, t1_adj)
    elif isinstance(seg, Line):
        p, q = seg.start, seg.end
        new_seg = Line(p * (1 - t0) + q * t0, p * (1 - t1) + q * t1)
    else:
        raise Exception(
            "Segment sent to trimSeg is neither Line object nor CubicBezier obect.")
    if t0 == 0:
        assert new_seg.start == seg.start
    if t1 == 1:
        assert new_seg.end == seg.end
    return new_seg


def cropPath(path, T0, T1):  ###TOL uses isclose
    #    path = parse_path(path2str(path)) ###DEBUG (maybe can remove if speed demands it)
    if T1 == 1:
        seg1 = path[-1]
        t_seg1 = 1
        i1 = len(path) - 1
    else:
        (t_seg1, seg1) = pathT2tseg(path, T1)
        if isclose(t_seg1, 0):
            i1 = (seg_index(path, seg1) - 1) % len(path)
            seg1 = path[i1]
            t_seg1 = 1
        else:
            i1 = seg_index(path, seg1)
    if T0 == 0:
        seg0 = path[0]
        t_seg0 = 0
        i0 = 0
    else:
        (t_seg0, seg0) = pathT2tseg(path, T0)
        if isclose(t_seg0, 1):
            i0 = (seg_index(path, seg0) + 1) % len(path)
            seg0 = path[i0]
            t_seg0 = 0
        else:
            i0 = seg_index(path, seg0)

    if T0 < T1 and i0 == i1:
        new_path = Path(trimSeg(seg0, t_seg0, t_seg1))
    else:
        new_path = Path(trimSeg(seg0, t_seg0, 1))
        if T1 == T0:
            raise Exception("T0=T1 in cropPath.")
        elif T1 < T0:  # T1<T0 must cross discontinuity case
            if not path.isclosed():
                raise Exception(
                    "T1<T0 and path is open.  I think that means you put in the wrong T values.")
            else:
                for i in range(i0 + 1, len(path)):
                    new_path.append(path[i])
                for i in range(0, i1):
                    new_path.append(path[i])
        else:  # T0<T1 straight-forward case
            for i in range(i0 + 1, i1):
                new_path.append(path[i])

        if t_seg1 != 0:
            new_path.append(trimSeg(seg1, 0, t_seg1))

    # ####check this path is put together properly DEBUG ONLY
    # #check end
    # path_at_T1 = path.point(T1)
    # if new_path[-1].end != path_at_T1:
    #     if isNear(new_path[-1].end,path_at_T1):
    #         new_path[-1].end = path_at_T1
    #     else:
    #         raise Exception("Cropped path doesn't end where it should.")
    # #check start
    # path_at_T0 = path.point(T0)
    # if new_path[0].start != path_at_T0:
    #     if isNear(new_path[0].start, path_at_T0):
    #         new_path[0].start = path_at_T0
    #     else:
    #         raise Exception("Cropped path doesn't start where it should.")
    # #check inner joints
    # for i in range(len(new_path)-1):
    #     if new_path[i].end != new_path[i+1].start:
    #         if isNear(new_path[i].end, new_path[i+1].start):
    #             new_path[i].end = new_path[i+1].start
    #         else:
    #             raise Exception("Cropped path doesn't start where it should.")
    return new_path


def boundingBox(curves):
    """input: Path or Line or CubicBezier or a list of such objects
    output: a 4-tuple of points for a box that bounds path
    """

    def segBoundingBox(seg):
        """
        Finds a BB containing a Line or CubicBezier"""
        if isinstance(seg, CubicBezier):
            cubpts = (seg.start, seg.control1, seg.control2, seg.end)
            xmin, xmax = cubicMinMax_x(cubpts)
            ymin, ymax = cubicMinMax_y(cubpts)
        elif isinstance(seg, Line):
            xmin = min(seg.start.real, seg.end.real)
            xmax = max(seg.start.real, seg.end.real)
            ymin = min(seg.start.imag, seg.end.imag)
            ymax = max(seg.start.imag, seg.end.imag)
        else:
            raise Exception("seg must be a Line or CubicBezier object")
        return (xmin, xmax, ymin, ymax)

    def bigBoundingBox(bbs):
        """Finds a BB containing a bunch of smaller bounding boxes"""
        xmins, xmaxs, ymins, ymaxs = zip(*bbs)
        xmin = min(xmins)
        xmax = max(xmaxs)
        ymin = min(ymins)
        ymax = max(ymaxs)
        return (xmin, xmax, ymin, ymax)

    if isinstance(curves, Line) or isinstance(curves, CubicBezier):
        return segBoundingBox(curves)
    elif isinstance(curves, Path):
        bbs = [segBoundingBox(seg) for seg in curves]
        return bigBoundingBox(bbs)
    elif all([isinstance(c, Line) or isinstance(c, CubicBezier) or isinstance(
            c, Path) for c in curves]):
        bblist = [boundingBox(c) for c in curves]
        return bigBoundingBox(bblist)
    else:
        raise Exception(
            "Input argument should be a Path or Line or CubicBezier or a list of such objects.")


def path2str(path, closure_tolerance=0):
    d = "M %s,%s" % (z2xy(path[0].start))
    for seg in path[0:len(path) - 1]:
        if isinstance(seg, CubicBezier):
            d += " C %s,%s %s,%s %s,%s" % (
            z2xy(seg.control1) + z2xy(seg.control2) + z2xy(seg.end))
        elif isinstance(seg, Line):
            d += " L %s,%s" % z2xy(seg.end)
        else:
            raise Exception(
                'Path segment is neither Line object nor CubicBezier object.')
    lastseg = path[len(path) - 1]
    if isinstance(lastseg, CubicBezier):
        d += " C %s,%s %s,%s %s,%s" % (
        z2xy(lastseg.control1) + z2xy(lastseg.control2) + z2xy(lastseg.end))
    elif isinstance(lastseg, Line):
        d += " L %s,%s" % z2xy(lastseg.end)
    else:
        raise Exception(
            'Path segment is neither Line object nor CubicBezier object.')
    if path.isclosed:
        if len(path) > 1 or not isinstance(lastseg, Line):
            d += "z"
    return d


def pasteablePath(path):
    return "parse_path('%s')" % path2str(path)


def disvg(paths, colors=[], nodes=None, node_colors=None, node_radii=None,
          lines=None, line_colors=None,
          filename=os_path.join(getcwd(), 'temporary_displaySVGPaths.svg'),
          openInBrowser=False, stroke_width=stroke_width_default,
          margin_size=0.1):
    from xml.dom.minidom import parse as md_xml_parse
    import svgwrite
    if os_path.dirname(filename) == '':
        filename = os_path.join(getcwd(), filename)
    if isinstance(paths, CubicBezier) or isinstance(paths, Line):
        paths = [paths]
    if colors == []:
        colors = ['blue'] * len(paths)
    else:
        assert len(colors) == len(paths)

    # set up the viewBox and background rectangle
    xmin, xmax, ymin, ymax = boundingBox(paths)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1
    if dy == 0:
        dy = 1
    xmin -= margin_size * dx + float(stroke_width) / 2
    ymin -= margin_size * dy + float(stroke_width) / 2
    dx += 2 * margin_size * dx + stroke_width
    dy += 2 * margin_size * dy + stroke_width
    vb = "%s %s %s %s" % (xmin, ymin, dx, dy)
    if dx > dy:
        szx = '600px'
        szy = str(round(float(600 * dy) / dx)) + 'px'
    else:
        szx = str(round(float(600 * dx) / dy)) + 'px'
        szy = '600px'

    # Create an SVG file containing Paths, Lines, and nodes (Circles of fixed radius)
    dwg = svgwrite.Drawing(filename=filename, size=(szx, szy), viewBox=vb)
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None,
                     fill='white'))  # add white background
    for i, p in enumerate(paths):
        if isinstance(p, Path):
            ps = path2str(p)
        elif isinstance(p, Line) or isinstance(p, CubicBezier):
            ps = path2str(Path(p))
        else:
            ps = p
        dwg.add(dwg.path(ps, stroke=colors[i], stroke_width=str(stroke_width),
                         fill='none'))
    # add lines
    if lines != None:
        if line_colors == None:
            line_colors = ["#0000ff"] * len(lines)
        for i, l in enumerate(lines):
            start = (l.start.real, l.start.imag);
            end = (l.end.real, l.end.imag)
            dwg.add(dwg.line(start=start, end=end, stroke=line_colors[i],
                             stroke_width=str(stroke_width), fill='none'))

    # add nodes
    if nodes != None:
        if node_colors == None:
            node_colors = ['purple'] * len(nodes)
        if node_radii == None:
            node_radii = [1] * len(nodes)
        for i_pt, pt in enumerate([(z.real, z.imag) for z in nodes]):
            dwg.add(dwg.circle(pt, node_radii[i_pt], stroke=node_colors[i_pt],
                               fill=node_colors[i_pt]))

    # save svg
    if not os_path.exists(os_path.dirname(filename)):  # debug folder
        makedirs(os_path.dirname(filename))
    dwg.save()

    xmlstring = md_xml_parse(filename).toprettyxml()
    with open(filename, 'w') as f:
        f.write(xmlstring)

    # try to open in web browser
    if openInBrowser and try_to_open_svgs_in_browser:
        try:
            openFileInBrowser(filename)
        except:
            print(
            "Failed to open output SVG in browser.  SVG saved to:\n%s" % filename)


def svgSlideShow(pathcolortuplelist, save_directory=None,
                 clear_directory=False, suppressOutput=False):
    # INPUT: a list of tuples (pathlist,colorlist)
    from andysmod import output2file

    # create save_directory (if doesn't already exist)
    if save_directory == None:
        save_directory = os_path.join(getcwd(), 'slideshow')
    if not os_path.exists(save_directory):  # debug folder
        makedirs(save_directory)

    if clear_directory:  # clear all files in slideshow folder
        for the_file in listdir(save_directory):
            file_path = os_path.join(save_directory, the_file)
            if os_path.isfile(
                    file_path):  # make sure file is a file (not subfolder)
                unlink(file_path)  # delete file

    # create svg for each image in slideshow
    for i, tup in enumerate(pathcolortuplelist):
        (paths, colors) = tup
        # make name for svg file
        if i < 10:
            indstr = '00' + str(i)
        elif i < 100:
            indstr = '0' + str(i)
        else:
            indstr = str(i)
        filename = os_path.join(save_directory, indstr + '.svg')

        # make svg file
        disvg(paths, colors, filename=filename)

    # create readme
    readme_mes = "If you're on Windows and have ImageMagick installed, an easy way to view this slideshow is the following:\nopen powershell in this folder and type\nmogrify -format jpg *.svg\nThis will convert all svg files to jpgs (alternatively just double click on 'convertSVGs2jpgs.bat')\nthen open the first one in Windows Photo Viewer and watch the slideshow\n"
    readme_filename = os_path.join(save_directory, "0000Readme.txt")
    output2file(readme_mes, filename=readme_filename, mode='w')

    # create a batch file which will convert svgs to jpgs using ImageMagick
    output2file('mogrify -format jpg *.svg',
                filename=os_path.join(save_directory, "convertSVGs2jpgs.bat"),
                mode='w')

    if not suppressOutput:  # tell user all is good.
        print("Done.  SVG files saved to\n%s" % save_directory)


@memoize
def ptInsideClosedPath(pt, outpt, path):
    """"returns true if pt is a point inside path.  outpt is a point you know
    is outside path."""
    assert path.isclosed()
    intersections = path.intersect(Line(pt, outpt))
    # intersections = pathXpathIntersections(Path(Line(pt, outpt)), path)
    if len(intersections) % 2:
        return True
    else:
        return False


def svg2pathlist(SVGfileLocation):
    doc = minidom.parse(SVGfileLocation)  # parseString also exists
    # Use minidom to extract path strings from input SVG
    path_strings = [p.getAttribute('d') for p in
                    doc.getElementsByTagName('path')]
    # Use minidom to extract polyline strings from input SVG, convert to path strings, add to list
    path_strings += [polylineStr2pathStr(p.getAttribute('points')) for p in
                     doc.getElementsByTagName('polyline')]
    # Use minidom to extract polygon strings from input SVG, convert to path strings, add to list
    path_strings += [polylineStr2pathStr(p.getAttribute('points')) + 'z' for p
                     in doc.getElementsByTagName('polygon')]
    # currently choosing to ignore line objects (assuming... all lines are fixes for non-overlapping mergers?)
    # Use minidom to extract line strings from input SVG, convert to path strings, and add them to list
    path_strings += ['M' + p.getAttribute('x1') + ' ' + p.getAttribute(
        'y1') + 'L' + p.getAttribute('x2') + ' ' + p.getAttribute('y2') for p
                     in doc.getElementsByTagName('line')]
    doc.unlink()
    return [parse_path(ps) for ps in path_strings]


    # def arclength(curve, tau0, tau1):
    #     """
    #     INPUT:
    #     pathORseg -  should be a CubicBezier, Line, of Path of CubicBezier
    #         and/or Line objects.
    #     T0_or_t0 and T1_or_t1  - should be floats between 0 and 1.
    #     OUTPUT: Returns the approximate arclength of pathORseg between T0_or_t0 and
    #         T1_or_t1
    #     """
    #     # if isinstance(pathORseg, Path):
    #     #     return cropPath(pathORseg, T0_or_t0, T1_or_t1).length()
    #     # else:
    #     #     return trimSeg(pathORseg, T0_or_t0, T1_or_t1).length()
    #     return curve.length(tau0, tau1)


    # def invArclength(pathORseg, s, subdivisions=1000000):
    #     """
    #     INPUT: pathORseg should be a CubicBezier, Line, of Path of CubicBezier
    #     and/or Line objects.
    #     OUTPUT: Returns a float, t, such that the arclength of pathORseg from 0 to
    #     t is approximately s.
    #     Note: To increase accuracy, increase subdivisions (default=1000).  See
    #         comment in svg.path.path.CubicBezer.length() method for more info.
    #     """
    # assert 0<= s <= pathORseg.length()
    # if isinstance(pathORseg,Path):
    #     path = pathORseg
    #     seg_lengths = [seg.length() for seg in path]
    #     lsum = 0
    #     # Find which segment the point we search for is located on
    #     for k, l in enumerate(seg_lengths):
    #         if lsum <= s <= lsum+l:
    #             return segt2PathT(path,path[k],invArclength(path[k],s-lsum))
    #         lsum+=l
    #     assert lsum+l< s <= path.length()
    #     return 1
    # elif isinstance(pathORseg,CubicBezier):
    #     cubic = pathORseg
    #     current_point = cubic.start
    #     lenght = 0
    #     delta = float(1)/subdivisions
    #     for x in range(1, subdivisions+1):
    #         t = delta*x
    #         next_point = cubic.point(t)
    #         distance = abs(next_point - current_point)
    #         lenght += distance
    #         current_point = next_point
    #         if lenght > s:
    #             return t
    #     else:
    #         return 1
    # elif isinstance(pathORseg,Line):
    #     line = pathORseg
    #     return float(s)/line.length()
    # else:
    #     raise Exception("First argument must be a CubicBezier object, Line object, or a Path object composed of such objects.")
