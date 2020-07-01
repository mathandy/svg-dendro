from __future__ import division
from numpy import isclose
from svgpathtools import Path, CubicBezier, Line, disvg, inv_arclength


def is_differentiable(path, tol=1e-8, return_kinks=False):
    for idx in range(len(path)):
        u = path[(idx-1)%len(path)].unit_tangent(1)
        v = path[idx].unit_tangent(0)
        u_dot_v = u.real*v.real + u.imag*v.imag
        if abs(u_dot_v - 1) > tol:
            return False
    return True


def kinks(path, tol=1e-8, return_kinks=False):
    """returns indices of segments that start on a non-differentiable joint."""
    kink_list = []
    for idx in range(len(path)):
        if idx == 0 and not path.isclosed():
            continue
        u = path[(idx - 1) % len(path)].unit_tangent(1)
        v = path[idx].unit_tangent(0)
        u_dot_v = u.real*v.real + u.imag*v.imag
        if abs(u_dot_v - 1) > tol:
            kink_list.append(idx)
    return kink_list


def _report_unfixable_kinks(_path,_kink_list):
    mes = ("\n%s kinks have been detected at that cannot be smoothed.\n"
           "To ignore these kinks and fix all others, run this function "
           "again with the second argument 'ignore_unfixable_kinks=True' "
           "The locations of the unfixable kinks are at the beginnings of "
           "segments: %s" % (len(_kink_list), _kink_list))
    disvg(_path, nodes=[_path[idx].start for idx in _kink_list])
    raise Exception(mes)


def smooth_joint(seg0, seg1, maxjointsize=3, tightness=1):
    """
    Input: two Line or CubicBezier objects seg0,seg1 such that
        seg0.end==seg1.start, and jointsize, a positive number

    Output: seg0_trimmed, elbow, seg1_trimmed, where elbow is a cubic bezier
        object that smoothly connects seg0_trimmed and seg1_trimmed.
    """
    assert seg0.end == seg1.start
    assert 0 < maxjointsize
    assert 0 < tightness < 2
#    sgn = lambda x:x/abs(x)
    q = seg0.end
    v = seg0.unit_tangent(1)
    w = seg1.unit_tangent(0)
    max_a = float(maxjointsize)/2
    a = min(max_a, min(seg1.length(), seg0.length()) / 20)
    if isinstance(seg0,Line) and isinstance(seg1,Line):
        '''
        Note: Letting
            c(t) = elbow.point(t), v= the unit tangent of seg0 at 1, w = the
            unit tangent vector of seg1 at 0,
            Q = seg0.point(1) = seg1.point(0), and a,b>0 some constants.
            The elbow will be the unique CubicBezier, c, such that
            c(0)= Q-av, c(1)=Q+aw, c'(0) = bv, and c'(1) = bw
            where a and b are derived above/below from tightness and
            maxjointsize.
        '''
#        det = v.imag*w.real-v.real*w.imag
        # Note:
        # If det is negative, the curvature of elbow is negative for all
        # real t if and only if b/a > 6
        # If det is positive, the curvature of elbow is negative for all
        # real t if and only if b/a < 2

#        if det < 0:
#            b = (6+tightness)*a
#        elif det > 0:
#            b = (2-tightness)*a
#        else:
#            raise Exception("seg0 and seg1 are parallel lines.")
        b = (2 - tightness)*a
        elbow = CubicBezier(q - a*v, q - (a - b/3)*v, q + (a - b/3)*w, q + a*w)
        seg0_trimmed = Line(seg0.start, elbow.start)
        seg1_trimmed = Line(elbow.end, seg1.end)
        return seg0_trimmed, [elbow], seg1_trimmed
    elif isinstance(seg0, Line) and isinstance(seg1, CubicBezier):
        '''
        Note: Letting
            c(t) = elbow.point(t), v= the unit tangent of seg0 at 1,
            w = the unit tangent vector of seg1 at 0,
            Q = seg0.point(1) = seg1.point(0), and a,b>0 some constants.
            The elbow will be the unique CubicBezier, c, such that
            c(0)= Q-av, c(1)=Q, c'(0) = bv, and c'(1) = bw
            where a and b are derived above/below from tightness and
            maxjointsize.
        '''
#        det = v.imag*w.real-v.real*w.imag
        # Note: If g has the same sign as det, then the curvature of elbow is
        # negative for all real t if and only if b/a < 4
        b = (4 - tightness)*a
#        g = sgn(det)*b
        elbow = CubicBezier(q - a*v, q + (b/3 - a)*v, q - b/3*w, q)
        seg0_trimmed = Line(seg0.start, elbow.start)
        return seg0_trimmed, [elbow], seg1
    elif isinstance(seg0, CubicBezier) and isinstance(seg1, Line):
        rseg1_trimmed, relbow, rseg0 = smooth_joint(seg1.reversed(),
                                                    seg0.reversed(),
                                                    maxjointsize=maxjointsize,
                                                    tightness=tightness)
        elbow = relbow[0].reversed()
        return seg0, [elbow], rseg1_trimmed.reversed()
    elif isinstance(seg0, CubicBezier) and isinstance(seg1, CubicBezier):
        # find a point on each seg that is about a/2 away from joint.  Make 
        # line between them.
        t0 = inv_arclength(seg0, seg0.length() - a/2)
        t1 = inv_arclength(seg1, a/2)
        seg0_trimmed = seg0.cropped(0, t0)
        seg1_trimmed = seg1.cropped(t1, 1)
        seg0_line = Line(seg0_trimmed.end, q)
        seg1_line = Line(q, seg1_trimmed.start)
        dummy, elbow0, seg0_line_trimmed = smooth_joint(seg0_trimmed, seg0_line)
        seg1_line_trimmed, elbow1, dummy = smooth_joint(seg1_line, seg1_trimmed)
        seg0_line_trimmed, elbowq, seg1_line_trimmed = smooth_joint(seg0_line_trimmed, seg1_line_trimmed)

        elbow = elbow0 + [seg0_line_trimmed] +  elbowq + [seg1_line_trimmed] + elbow1
        return seg0_trimmed, elbow, seg1_trimmed
    else:
        raise TypeError("The first two arguments must each be eith a Line or "
                        "CubicBezier object.")
        
        
def smooth_path(path, ignore_unfixable_kinks=False):
    """returns a path with no non-differentiable joints."""
    if len(path) == 1:
        return path
        
    sharp_kinks = []
    new_path = [path[0]]
    for idx in range(len(path)):
        if idx == len(path)-1:
            if not path.isclosed():
                continue
            else:
                seg1 = new_path[0]
        else:
            seg1 = path[idx + 1]
        seg0 = new_path[-1]

        unit_tangent0 = seg0.unit_tangent(1)
        unit_tangent1 = seg1.unit_tangent(0)
        
        if isclose(unit_tangent0, unit_tangent1):  # joint is already smooth
            if idx != len(path)-1:
                new_path.append(seg1)
            continue
        else:
            kink_idx = idx + 1  # kink at start of this seg
            if isclose(-unit_tangent0, unit_tangent1):
                # joint is sharp 180 deg (must be fixed manually)
                new_path.append(seg1)
                sharp_kinks.append(kink_idx)
            else:  # joint is not smooth, let's  smooth it.
                new_seg0, elbow_segs, new_seg1 = smooth_joint(seg0, seg1)
                new_path[-1] = new_seg0
                new_path += elbow_segs
                if idx == len(path) - 1:
                    new_path[0] = new_seg1
                else:
                    new_path.append(new_seg1)

    # If unfixable kinks were found, let the user know
    if sharp_kinks and not ignore_unfixable_kinks:
        _report_unfixable_kinks(path, sharp_kinks)

    return Path(*new_path)
    





# from svgpathtools import svg2pathlist
#
# #test
# ignoreUnfixableKinks = False
# stroke_width = 1
# file_loc = "/Users/Andy/Google Drive/TRrelated/TestSVGs3ringaroundcore/kinktest3.svg"
# # file_loc = "/Users/Andy/Google Drive/TRrelated/TestSVGs3ringaroundcore/kinktest3-problempath.svg"
# pathlist = svg2pathlist(file_loc)
# for path in pathlist:
#     for i in range(len(path)):
#         assert path[i].end==path[(i+1)%len(path)].start
#         assert isinstance(path[i],Line) or isinstance(path[i],CubicBezier)
# rpathlist = [p.reversed() for p in pathlist]
# for path in rpathlist:
#     for i in range(len(path)):
#         assert path[i].end == path[(i + 1) % len(path)].start
#         assert isinstance(path[i],Line) or isinstance(path[i],CubicBezier)
# smoothedpathlist = [smooth_path(path, ignore_unfixable_kinks=ignoreUnfixableKinks) for path in pathlist]
# rsmoothedpathlist = [smooth_path(path, ignore_unfixable_kinks=ignoreUnfixableKinks) for path in rpathlist]

#disvg(pathlist)
#
#disvg(smoothedpathlist)
#
#disvg(rpathlist)
#
#disvg(rsmoothedpathlist)

#
# print len(pathlist)
# print len(smoothedpathlist)
# print len(rpathlist)
# print len(rsmoothedpathlist)
# for k, p in enumerate(smoothedpathlist):
#     res1 = is_differentiable(pathlist[k])
#     res2 = is_differentiable(p)
#     closure_check = p.isclosed() == pathlist[k].isclosed()
#     c2 = (pathlist[k].isclosed(), p.isclosed())
#     print res1, res2, "Closure check passed =", closure_check, c2
#     if not res2:
#         print kinks(p)
#         ncols = ['red']*len(kinks(p)) + ['green']*2
#         disvg(p, nodes=[p[k].start for k in kinks(p)] + [p.start, p.end],
#                         node_colors = ncols)
#     if not closure_check:
#         disvg(pathlist[k], nodes=[pathlist[k].start, pathlist[k].end])
#         disvg(p, nodes=[p.start, p.end], node_colors=['green', 'red'])
#
#
# for k, p in enumerate(rsmoothedpathlist):
#     res1 = is_differentiable(rpathlist[k])
#     res2 = is_differentiable(p)
#     closure_check = p.isclosed() == rpathlist[k].isclosed()
#     c2 = (rpathlist[k].isclosed(), p.isclosed())
#     print res1, res2, "Closure check passed =", closure_check, c2
#     if not res2:
#         print kinks(p)
#         ncols = ['red']*len(kinks(p)) + ['green']*2
#         disvg(p, nodes=[p[k].start for k in kinks(p)] + [p.start, p.end],
#                         node_colors = ncols)
#
#     if not closure_check:
#         disvg(rpathlist[k], nodes=[rpathlist[k].start, rpathlist[k].end])
#         disvg(p, nodes=[p.start, p.end], node_colors=['green', 'red'])

#from svg2rings import svg2rings
##test 1
#file_loc = "C:\\Users\\Andy\\Desktop\\TRrelated\\TestSVGs3ringaroundcore\\kinktest_allLines.svg"
#center,ringlist = svg2rings(file_loc)
#pathlist = [r.path for r in ringlist]
##pathlist = svg2pathlist(file_loc)
#dis(pathlist,stroke_width=1)
#pause(2)
#smoothedpathlist = [smooth_path(path) for path in pathlist]
#dis(smoothedpathlist,stroke_width=1)
#pause(2)
#
##test 1CW
#from andysSVGpathTools import reversePath
#pathlist = [reversePath(r.path) for r in ringlist]
##pathlist = svg2pathlist(file_loc)
#dis(pathlist,stroke_width=1)
#pause(2)
#smoothedpathlist = [smooth_path(path) for path in pathlist]
#dis(smoothedpathlist,stroke_width=1)
#pause(2)
#
##test 2
#file_loc = "C:\\Users\\Andy\\Desktop\\TRrelated\\TestSVGs3ringaroundcore\\kinktest_curvy.svg"
#center,ringlist = svg2rings(file_loc)
#pathlist = [r.path for r in ringlist]
##pathlist = svg2pathlist(file_loc)
#dis(pathlist,stroke_width=1)
#pause(2)
#smoothedpathlist = [smooth_path(path) for path in pathlist]
#dis(smoothedpathlist,stroke_width=1)