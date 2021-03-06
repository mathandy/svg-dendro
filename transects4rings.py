# External Dependencies
from __future__ import division
from math import sqrt
from numpy import poly1d
from warnings import warn
import os
from svgpathtools import (Path, Line, polyroots, real,
                          imag, disvg, wsvg)
from svgpathtools.misctools import isclose
poly_imag_part = imag
poly_real_part = real

# Internal Dependencies
from andysSVGpathTools import pathT2tseg
import options4rings as opt

disvg = disvg if opt.try_to_open_svgs_in_browser else wsvg


def isPointOutwardOfSeg(pt, seg):
    """
    Let c(t) = seg.point(t). If <pt-c(t),c'(t)>==0 has a solution 0<=t<=1,
    then this the normal leaving seg at t will intersect with pt.
    """
    c = seg.poly()
    u = poly1d((pt,)) - c
    dc = c.deriv()
    inner_prod = (poly_real_part(u)*poly_real_part(dc) +
                  poly_imag_part(u)*poly_imag_part(dc))

    # Note about nondeg_cond:
    # The derivative of a CubicBezier object can be zero, but only at
    # at most one point.  For example if seg.start==seg.control1, then (in
    # theory) seg.derivative(t)==0 if and only if t==0.
    def dot_prod(z1, z2):
        return (z1.real*z2.real + 
                z1.imag*z2.imag)

    def allcond(_t):
        from andysSVGpathTools import segUnitTangent
        tt = max(0, min(1, _t.real))
        lin2pt = Line(seg.point(tt), pt)
        real_cond = isclose(_t.imag, 0)
        bezier_cond = (0 < _t.real < 1 or
                       isclose(_t.real, 0) or
                       isclose(_t.real, 1))
        nondeg_cond = (not isclose(seg.derivative(tt), 0) or 
                       isclose(0, dot_prod(segUnitTangent(seg, tt),
                                           lin2pt.unit_tangent())))
        outward_cond = \
            0 < dot_prod(lin2pt.unit_tangent(), -1j*segUnitTangent(seg, tt))
        
        return real_cond and bezier_cond and nondeg_cond and outward_cond

    inward_tvals = polyroots(inner_prod, condition=allcond)
    return [tval.real for tval in inward_tvals]


def isPointOutwardOfPath(pt, path, outerRing=None, justone=False):
    """returns a list of (seg_idx, t) tuples s.t. the outward normal to
    seg at t intersects pt.
    if justone=True, then this list will be of length 1
    note: outerRing is only needed incase path contains corners."""
    inward_segt_pairs = []
    for idx, seg in enumerate(path):
        tvals = isPointOutwardOfSeg(pt, seg)
        if justone and len(tvals):
            return [(idx,tvals[0])]
        for t in tvals:
            inward_segt_pairs.append((idx, t))
    return inward_segt_pairs


def invTransect(T, sorted_ring_list, warnifnotunique=True):
    """Finds a transect that ends at T.

    In the case there are more than one, if warnifnotunique=True,
    user will be warned, but this may slow down transect generation.
    IS `warnifnotunique` IMPLEMENTED? Maybe not ...

    Returns:
        list of tuples of form [(pt, ring_idx, seg_idx, t), ...]
    """
    cur_ring = sorted_ring_list[-1]
    cur_idx = len(sorted_ring_list) - 1
    init_t, init_seg = pathT2tseg(cur_ring.path, T)
    init_seg_idx = cur_ring.path.index(init_seg)
    transect_info = [(cur_ring.point(T),
                      len(sorted_ring_list) - 1,
                      init_seg_idx,
                      init_t)]
    cur_pt = transect_info[-1][0]

    while cur_idx > 0:

        # Find all rings this transect segment could be coming from
        test_rings = []
        for r_idx, r in list(enumerate(sorted_ring_list[:cur_idx]))[::-1]:
            test_rings.append((r_idx, r))
            if r.path.isclosed():
                break

        test_ring_results = []
        for r_idx, test_ring in test_rings:
            args = (cur_pt, test_ring.path, cur_ring)
            inward_segt_list = isPointOutwardOfPath(*args, justone=False)

            for seg_idx, t in inward_segt_list:
                test_ring_results.append((r_idx, seg_idx, t))

        # sort choices by distance to cur_pt
        def dist(res_):
            r_idx_, seg_idx_, t_ = res_
            new_pt_ = sorted_ring_list[r_idx_].path[seg_idx_].point(t_)
            return abs(cur_pt - new_pt_)
        sorted_results = sorted(test_ring_results, key=dist)

        # Find the closest result such that the transect does not go through
        # any other rings on it's way to cur_pt
        for res in sorted_results:
            wr_idx, wseg_idx, wt = res
            new_pt = sorted_ring_list[wr_idx].path[wseg_idx].point(wt)
            tr_line = Line(new_pt, cur_pt)

            winner = not any(r.path.intersect(tr_line)
                             for ri, r in test_rings if ri != wr_idx)
            if winner:
                break
        else:
            if opt.skip_transects_that_dont_exist:
                bdry_ring = sorted_ring_list[-1]
                s_rel = bdry_ring.path.length(T1=T) / bdry_ring.path.length()
                fn = sorted_ring_list[0].svgname + \
                     "_partial_transect_%s.svg" % s_rel
                fn = os.path.join(opt.output_directory, fn)
                wsvg([r.path for r in sorted_ring_list],
                     nodes=[tr[0] for tr in transect_info],
                     filename=fn)
                warn("\nNo transect exists ending at relative arc "
                     "length %s.  An svg displaying this partial transect has"
                     "been saved to:\n%s\n" % (s_rel, fn))
                return []
            elif opt.accept_transect_crossings:
                wr_idx, wseg_idx, wt = sorted_results[0]
            else:
                disvg([r.path for r in sorted_ring_list],
                      nodes=[tr[0] for tr in transect_info]) # DEBUG line
                bdry_ring = sorted_ring_list[-1]
                s_rel = bdry_ring.path.length(T1=T) / bdry_ring.path.length()
                raise Exception("No transect exists ending at relative arc "
                                "length %s." % s_rel)

        # Record the closest choice
        transect_info.append((sorted_ring_list[wr_idx].path[wseg_idx].point(wt),
                              cur_idx,
                              wseg_idx,
                              wt))
        cur_ring = sorted_ring_list[wr_idx]
        cur_pt = transect_info[-1][0]
        cur_idx = wr_idx

        # Erroneous Termination
        if cur_idx < 0 and sorted_ring_list.index(cur_ring) != 0:
            disvg([r.path for r in sorted_ring_list],
                  nodes=[tr[0] for tr in transect_info]) # DEBUG line
            bdry_ring = sorted_ring_list[-1]
            s_rel = bdry_ring.path.length(T1=T) / bdry_ring.path.length()
            raise Exception("Something went wrong finding inverse transect at "
                            "relative arc length %s." % s_rel)
    return transect_info


def generate_inverse_transects(ring_list, Tvals):
    """The main purpose of this function is to run invTransect for all Tvals 
    and format the data"""
    ring_sorting, sorted_ring_list = \
        zip(*sorted(enumerate(ring_list), key=lambda i_r: i_r[1].sort_index))

    def unsorted_index(idx):
        return ring_sorting[idx]

    data = []
    data_indices = []
    skipped_angle_indices = []
    for T_idx, T in enumerate(Tvals):
        tran_info = invTransect(T, sorted_ring_list, opt.warn_if_not_unique)

        if not tran_info and opt.skip_transects_that_dont_exist:
            skipped_angle_indices.append(T_idx)
            continue
        transect = []
        transect_rings = []
        for pt, ring_idx, seg_idx, t in tran_info:
            transect.append(pt)
            transect_rings.append(unsorted_index(ring_idx))
        transect.append(sorted_ring_list[0].center)
        transect_rings.append('core')
        transect.reverse()
        transect_rings.reverse()
        data.append(transect)
        data_indices.append(transect_rings)
    return data, data_indices, skipped_angle_indices


def generate_unsorted_transects(ring_list, center):
    from options4rings import (
        basic_output_on, warnings_output_on, N_transects,
        unsorted_transect_debug_output_folder, unsorted_transect_debug_on,
        colordict)
    from misc4rings import (
        transect_from_angle, normalLineAt_t_toInnerSeg_intersects_withOuter)
    from andysSVGpathTools import pathlistXlineIntersections
    from andysmod import Timer
    import operator
    from random import uniform

    # Find outer boundary ring
    for r in ring_list:
        if r.color == colordict['boundary']:
            boundary_ring = r
            break
    else:
        warnings_output_on.dprint(
            "[Warning:] Having trouble finding outer boundary - it "
            "should be color %s.  Will now search for a ring of a "
            "similar color and if one is found, will use that.\n"
            "" % colordict['boundary'])
        from misc4rings import closestColor
        for r in ring_list:
            if colordict['boundary'] == closestColor(r.color , colordict):
                boundary_ring = r
                basic_output_on.dprint(
                    "Found a ring of color %s, using that one." % r.color)
                break
        else:
            warnings_output_on.dprint(
                "[Warning:] Outer boundary could not be found by color "
                "(or similar color).  This is possibly caused by the "
                "outer boundary ring not being closed - in this case "
                "you'd be able to see a (possibly quite small) gap "
                "between it's startpoint and endpoint. Using the ring "
                "of greatest maximum radius as the boundary ring (and "
                "hoping if there is a gap none of the transects hit it).\n")
            boundary_ring = max(ring_list, key=lambda x: x.maxR)

    # Find transects
    from time import time as current_time
    from andysmod import format_time
    tr_gen_start_time = current_time()
    data = []
    data_indices = []
    angles = []
    for k in range(N_transects):
        # estimate time remaining
        if k != 0:
            total_elapsed_time = current_time() - tr_gen_start_time
            estimated_time_remaining = (N_transects - k)*total_elapsed_time/k
            timer_str = 'Transect %s of %s || ' \
                        'Est. Remaining Time = %s || ' \
                        'Elapsed Time = %s' \
                        '' % (k+1, N_transects,
                              format_time(estimated_time_remaining),
                              format_time(total_elapsed_time))
            overwrite_progress = True
        else:
            timer_str = 'transect %s of %s' % (k+1, N_transects)
            overwrite_progress = False
            print('')

        # generate current transect
        with Timer(timer_str, overwrite=overwrite_progress):
            if unsorted_transect_debug_on:
                print('')
            test_angle = uniform(0, 1)
#                        test_angle = 0.408
            angles.append(test_angle)
            transect = [center]
            transect_rings = ['core']

            # used to keep track of which rings I've used and thus don't
            # need to be checked in the future
            unused_ring_indices = range(len(ring_list))

            # Find first transect segment (from core/center)
            # normal line to use to find intersections (from center to boundary ring)
            nl2bdry, seg_outer, t_outer = transect_from_angle(
                test_angle, center, boundary_ring.path, 'debug')

            # make normal line a little longer
            nl2bdry = Line(nl2bdry.start,
                           nl2bdry.start + 1.5*(nl2bdry.end-nl2bdry.start))

            tmp = pathlistXlineIntersections(
                nl2bdry, [ring_list[i].path for i in unused_ring_indices])
            tl, path_index, seg, tp = min(tmp, key=operator.itemgetter(0))

            transect.append(nl2bdry.point(tl))
            transect_rings.append(unused_ring_indices[path_index])
            del unused_ring_indices[path_index]

            # now for the rest of the transect
            num_rings_checked = 0
            while (ring_list[transect_rings[-1]] != boundary_ring and 
                   num_rings_checked < len(ring_list)):  # < is correct, already did first
                num_rings_checked += 1
                inner_path = ring_list[transect_rings[-1]].path
                inner_t = tp
                inner_seg = seg
                
                # normal line to use to find intersections (from center to boundary ring)
                nl2bdry, seg_outer, t_outer = \
                    normalLineAt_t_toInnerSeg_intersects_withOuter(
                        inner_t, inner_seg, boundary_ring.path, center, 'debug')

                # make normal line a little longer
                nl2bdry = Line(nl2bdry.start,
                               nl2bdry.start + 1.5*(nl2bdry.end-nl2bdry.start))
                
                normal_line_intersections = pathlistXlineIntersections(
                    nl2bdry, [ring_list[i].path for i in unused_ring_indices])

                try:
                    tl, path_index, seg, tp = min(normal_line_intersections,
                                                  key=operator.itemgetter(0))
                except ValueError:
                    raise
                if unsorted_transect_debug_on:
                    from andysmod import format001
                    inner_path_index = transect_rings[-1]
                    used_ring_paths = \
                        [r.path for i, r in enumerate(ring_list)
                         if i not in unused_ring_indices + [inner_path_index]]
                    used_ring_colors = ['black']*len(used_ring_paths)
                    unused_ring_paths = \
                        [ring_list[i].path for i in unused_ring_indices]
                    unused_ring_colors = \
                        [ring_list[i].color for i in unused_ring_indices]
                    transect_so_far = \
                        Path(*[Line(transect[i-1], transect[i])
                               for i in range(1, len(transect))])
                    paths = used_ring_paths + unused_ring_paths + \
                        [transect_so_far] + [inner_path] + [nl2bdry]
                    colors = used_ring_colors + unused_ring_colors + \
                        ['green'] + ['blue'] + ['black']

                    nodes_so_far = transect[1:-1]
                    potential_nodes = \
                        [nl2bdry.point(tltmp)
                         for tltmp, _, _, _ in normal_line_intersections]
                    nodes = nodes_so_far + potential_nodes
                    node_colors = ['red']*len(nodes_so_far) + \
                                  ['purple']*len(potential_nodes)
                    save_name = unsorted_transect_debug_output_folder + \
                                'unsorted_transect_debug_%s.svg' \
                                '' % format001(3, len(transect))
                    disvg(paths, colors, nodes=nodes, node_colors=node_colors,
                          center=center, filename=save_name, openInBrowser=False)
                    print("Done with %s out of (at most) %s transect "
                          "segments" % (len(transect), len(ring_list)))
                transect.append(nl2bdry.point(tl))
                transect_rings.append(unused_ring_indices[path_index])
                del unused_ring_indices[path_index]
            data.append(transect)
            data_indices.append(transect_rings)
    return data, data_indices, angles


def generate_sorted_transects(ring_list, center, angles2use=None):
    from options4rings import basic_output_on, N_transects
    from misc4rings import (
        transect_from_angle, normalLineAt_t_toInnerSeg_intersects_withOuter)
    from andysSVGpathTools import pathlistXlineIntersections
    from andysmod import Timer, format_time
    from svgpathtools import Line
    from random import uniform
    from time import time as current_time
    from operator import itemgetter

    tmp = sorted(enumerate(ring_list), key=lambda tup: tup[1].sort_index)
    ring_sorting, sorted_ring_list = zip(*tmp)

    def unsorted_index(idx):
        return ring_sorting[idx]

    # Find transects
    tr_gen_start_time = current_time()
    data = []
    data_indices = []
    angles = []
    for k in range(N_transects):
        # estimate time remaining
        if k != 0:
            total_elapsed_time = current_time() - tr_gen_start_time
            estimated_time_remaining = \
                (N_transects - k)*total_elapsed_time/k
            timer_str = 'Transect %s of %s || ' \
                        'Est. Remaining Time = %s || ' \
                        'Elapsed Time = %s' \
                        '' % (k+1, N_transects,
                              format_time(estimated_time_remaining),
                              format_time(total_elapsed_time))
            overwrite_progress = True
        else:
            timer_str = 'transect %s of %s' % (k+1, N_transects)
            overwrite_progress = False
            print('')

        # generate current transect
        with Timer(timer_str, overwrite=overwrite_progress):
            if angles2use:
                test_angle = angles2use[k]
            else:
                test_angle = uniform(0, 1)
            angles.append(test_angle)
            transect = [center]
            transect_rings = ['core']
            
            # find first (innermost) closed ring
            next_closed_ring = next(r for r in sorted_ring_list if r.isClosed())
            next_closed_ring_sidx = next_closed_ring.sort_index

            # Find first transect segment (from core/center)
            # Start by finding line that leaves center at angle and goes to 
            # the first closed ring
            nl2bdry, seg_outer, t_outer = transect_from_angle(
                test_angle, center, next_closed_ring.path, 'debug')

            # Make normal line a little longer
            end2use = nl2bdry.start + 1.5*(nl2bdry.end - nl2bdry.start)
            nl2bdry = Line(nl2bdry.start, end2use) 
            pot_paths = \
                [r.path for r in sorted_ring_list[0: next_closed_ring_sidx + 1]]
                
            # Note: intersections returned as (tl, path_index, seg, tp)
            pot_path_inters = pathlistXlineIntersections(nl2bdry, pot_paths) 
            tl, path_index, seg, tp = min(pot_path_inters, key=itemgetter(0))

            # updates
            transect.append(nl2bdry.point(tl))
            transect_rings.append(unsorted_index(path_index))
            cur_pos_si = path_index
            next_closed_ring = next(r for r in sorted_ring_list 
                                       if (r.sort_index > cur_pos_si and 
                                           r.isClosed()))
            next_closed_ring_sidx = next_closed_ring.sort_index

            # now for the rest of the transects
            # note: < is correct, already did first
            num_rings_checked = 0
            while (cur_pos_si < len(ring_list) - 1 and 
                   num_rings_checked < len(ring_list)):
                num_rings_checked += 1
                inner_t = tp
                inner_seg = seg
                
                # Find outwards normal line from current position to the next 
                # closed ring
                nl2bdry, seg_outer, t_outer = \
                    normalLineAt_t_toInnerSeg_intersects_withOuter(
                        inner_t, inner_seg, next_closed_ring.path, center, 'debug')

                # Make the normal line a bit longer to avoid numerical error
                end2use = nl2bdry.start + 1.5*(nl2bdry.end - nl2bdry.start)
                nl2bdry = Line(nl2bdry.start, end2use)

                pot_paths = [r.path for r in
                             sorted_ring_list[cur_pos_si + 1:
                                              next_closed_ring_sidx + 1]]
                tl, path_index, seg, tp = \
                    min(pathlistXlineIntersections(nl2bdry, pot_paths),
                        key=itemgetter(0))

                # updates
                transect.append(nl2bdry.point(tl))
                cur_pos_si += path_index + 1
                transect_rings.append(unsorted_index(cur_pos_si))
                if cur_pos_si < len(ring_list)-1:
                    next_closed_ring = next(r for r in sorted_ring_list
                                            if (r.sort_index > cur_pos_si and
                                                r.isClosed()))
                    next_closed_ring_sidx = next_closed_ring.sort_index

            data.append(transect)
            data_indices.append(transect_rings)
    return data, data_indices, angles


def save_transect_data(outputFile_transects, ring_list, data, data_indices,
                       angles, skipped_angles):
    from options4rings import basic_output_on
    with open(outputFile_transects,"wt") as out_file:
        out_file.write("transect angle, number of rings counted by transect, "
                       "distance time series... total number of svg paths = "
                       "%s\n" % (len(ring_list) + 1))
        for k in range(len(data)):
            distances = [abs(data[k][i+1]-data[k][i])
                         for i in range(len(data[k]) - 1)]

            # number of rings counted (i.e. number of little lines this
            # transect is made of)
            rings_counted_by_transect = len(distances)
            
            # Convert data_indices to sort_index indices
            indices = []
            for idx in data_indices[k]:
                if isinstance(idx, str):  # 'core' case
                    indices.append(idx)
                else:
                    indices.append(ring_list[idx].sort_index)

            row = str([angles[k]] + [rings_counted_by_transect] + distances)
            row = row[1:len(row)-1]
            row2 = str([angles[k]] + ['NA'] + indices)
            row2 = row2[1:len(row2)-1]
            out_file.write(row + '\n' + row2 + '\n')

        for angle in skipped_angles:
            row = str([angle] + ['skipped'])[1:-1]
            row2 = str([angle] + ['NA'] + ['NA'])[1:-1]
            out_file.write(row + '\n' + row2 + '\n')

    basic_output_on.dprint("Data from %s transects saved to:" % len(data))
    basic_output_on.dprint(outputFile_transects)


def save_transect_summary(outputFile_transect_summary, ring_list, data,
                          data_indices, angles):
    """records averages of all transect distances going from ring1 to ring2"""
    from options4rings import basic_output_on
    from andysmod import eucnormalize_numpy, flattenList
    num_transects = len(angles)

    def collect(some_dict,key,val):
        try:
            some_dict.update({key :[val]+some_dict[key]})
        except KeyError:
            some_dict.update({key : [val]})

    distance_time_series = [[abs(tr[i] - tr[i-1]) for i in range(1, len(tr))] 
                            for tr in data]
    
    arrow_guides = [[(tr[i-1],tr[i]) for i in range(1,len(tr))]
                    for tr in data_indices]

    def g(ridx):
        if isinstance(ridx, str):
            return ridx
        else:
            return ring_list[ridx].sort_index
    sorted_arrow_guides = [[(g(x), g(y)) for (x, y) in aguide] 
                           for aguide in arrow_guides]
                        
    def normalized(vec):
        mag = sqrt(sum(x*x for x in vec))
        return [x/mag for x in vec]
    
    normalized_distance_time_series = [normalized(tr) for tr in distance_time_series]

    # initialize some dictionaries (there names explain them)
    raw_distances_for_arrow = dict(); normalized_distances_for_arrow = dict() #arrow -> list of distances (without zeros)

    # put data in the above dictionaries
    [collect(raw_distances_for_arrow, *item) for item in
     zip(flattenList(arrow_guides), flattenList(distance_time_series))]

    [collect(normalized_distances_for_arrow, *item) for item in
     zip(flattenList(arrow_guides), flattenList(normalized_distance_time_series))]

    # initialize some more dictionaries (there names explain them)
    # arrow -> list of distances (without zeros)
    raw_distance_average_for_arrow_woZeros = dict()
    normalized_distance_average_for_arrow_woZeros = dict() 
    # arrow -> list of distances (with zeros)
    raw_distance_average_for_arrow_wZeros = dict()
    normalized_distance_average_for_arrow_wZeros = dict()

    # put data in the above dictionaries
    [[raw_distance_average_for_arrow_woZeros.update(
        {arrow: sum(dlist)/len(dlist)}) for arrow, dlist in
        raw_distances_for_arrow.items()]]

    # note: len(angles) = the number of transects generated
    [[raw_distance_average_for_arrow_wZeros.update(
        {arrow: sum(dlist)/num_transects}) for arrow, dlist in
        raw_distances_for_arrow.items()]]

    [[normalized_distance_average_for_arrow_woZeros.update(
        {arrow:sum(dlist)/len(dlist)}) for arrow, dlist in
        normalized_distances_for_arrow.items()]]

    # note: len(angles) = the number of transects generated
    [[normalized_distance_average_for_arrow_wZeros.update(
        {arrow:sum(dlist)/num_transects}) for arrow, dlist in
        normalized_distances_for_arrow.items()]]

    # in order to display averaged time series in order of length...
    # here's the permutation
    keyfcn = lambda k: len(distance_time_series[k])
    order2printTimeSeries = \
        sorted(range(num_transects), key=keyfcn, reverse=True)

    def deliminate_and_write(out_file,*args):
        """insert delimeters between args and output string"""
        output = str(args[0]).replace(',', ';')
        for k in range(len(args)):
            output += ', '+str(args[k]).replace(',', ';')
        out_file.write(output + '\n')

    with open(outputFile_transect_summary, "wt") as out_file:
        out_file.write("Number of path objects counted in SVG: %s\n"
                       "" % len(ring_list))
        out_file.write("Max number of rings counted by a transect: %s\n\n"
                       "" % (max([len(transect) for transect in data]) - 1))
        out_file.write("angle arrow guide is based on, type of timeseries, "
                       "time series..."+'\n')

        # record data without zeros
        for k in order2printTimeSeries:
            angle = angles[k]
            arrow_guide = arrow_guides[k]
            sorted_arrow_guide = sorted_arrow_guides[k]
            closure_guide = \
                [('core', ring_list[arrow_guide[0][1]].isApproxClosedRing())] + \
                [(ring_list[i].isApproxClosedRing(),
                  ring_list[j].isApproxClosedRing())
                 for i, j in arrow_guide[1: len(arrow_guide)]]

            raw_woZeros = [raw_distance_average_for_arrow_woZeros[arrow]
                           for arrow in arrow_guide]

            raw_wZeros = [raw_distance_average_for_arrow_wZeros[arrow]
                          for arrow in arrow_guide]

            normalized_woZeros = \
                [normalized_distance_average_for_arrow_woZeros[arrow]
                 for arrow in arrow_guide]

            normalized_wZeros = \
                [normalized_distance_average_for_arrow_wZeros[arrow]
                 for arrow in arrow_guide]

            deliminate_and_write(
                out_file, angle, 'arrow guide', *sorted_arrow_guide)
            deliminate_and_write(
                out_file, angle, 'closure', *closure_guide)
            deliminate_and_write(
                out_file, angle, 'raw w/o zeros', *raw_woZeros)
            deliminate_and_write(
                out_file, angle, 'raw with zeros', *raw_wZeros)
            deliminate_and_write(
                out_file, angle, 'normalized w/o zeros', *normalized_woZeros)
            deliminate_and_write(
                out_file, angle, 'normalized with zeros', *normalized_wZeros)

            deliminate_and_write(
                out_file, angle, 'renormalized raw w/o zeros',
                *eucnormalize_numpy(raw_woZeros))
            deliminate_and_write(
                out_file, angle, 'renormalized raw with zeros',
                *eucnormalize_numpy(raw_wZeros))
            deliminate_and_write(
                out_file, angle, 'renormalized normalized w/o zeros',
                *eucnormalize_numpy(normalized_woZeros))
            deliminate_and_write(
                out_file, angle, 'renormalized normalized with zeros',
                *eucnormalize_numpy(normalized_wZeros))
            out_file.write("\n")

    basic_output_on.dprint("Summary of transect results saved to:")
    basic_output_on.dprint(outputFile_transect_summary)
