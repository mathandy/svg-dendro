# External Dependencies
from __future__ import division, absolute_import, print_function
import os
from time import time as current_time
from warnings import warn
from svgpathtools import (parse_path, Path, Line, disvg, wsvg, kinks,
                          smoothed_path, bezier_segment)

# Internal Dependencies
from andysmod import format_time, inputyn
from misc4rings import pathXpathIntersections
from andysSVGpathTools import ptInsideClosedPath
from transects4rings import isPointOutwardOfPath
import options4rings as opt


def crop_to_unit_interval(tval, tol=opt.tol_intersections):
    """Ensure tval inside unit interval, [0, 1].

    If outside unit interval, but within `tol` of 0 or 1,
    set to 0 or 1 respectively.
    """
    assert 0 <= tval <= 1 or abs(tval) < tol or abs(tval - 1) < tol
    if tval <= 0:
        return 0
    elif tval >= 1:
        return 1
    else:
        return tval


def fix_svg(ring_list, center, svgname):
    """Check for human errors and create more perfect SVG"""

    # Discard inappropriately short rings
    opt.basic_output_on.dprint("\nChecking for inappropriately short "
                               "rings...", 'nr')
    tmp_len = len(ring_list)
    short_rings = [idx for idx, ring in enumerate(ring_list) if
                   ring.path.length() < opt.appropriate_ring_length_minimum]
    opt.basic_output_on.dprint("Done (%s inappropriately short rings "
                               "found)." % len(short_rings))
    if short_rings:
        if opt.create_svg_highlighting_inappropriately_short_rings:
            opt.basic_output_on.dprint("\nCreating svg highlighting "
                                       "inappropriately short rings...", 'nr')
            paths = [parse_path(r.string) for r in ring_list]
            colors = [r.color for r in ring_list]
            nodes = [ring_list[idx].path.point(0.5) for idx in short_rings]
            center_line = [Line(center-1, center+1)]

            shortrings_svg_filename = os.path.join(
                opt.output_directory, svgname + "_short-rings.svg")
            disvg(paths + [center_line], colors + [opt.colordict['center']],
                  nodes=nodes, filename=shortrings_svg_filename)

            args = opt.appropriate_ring_length_minimum, shortrings_svg_filename
            mes = ("Done.  SVG created highlighting short rings by placing a "
                   "node at each short ring's midpoint.  Note: since these "
                   "rings are all under {} pixels in length, they may be hard "
                   "to see and may even be completely covered by the node.  "
                   "SVG file saved to:\n{}").format(*args)
            opt.basic_output_on.dprint(mes)

        if opt.dont_remove_closed_inappropriately_short_rings:
            shortest_ring_length = min([r.path.length() for r in
                                        [ring_list[k] for k in short_rings if
                                         ring_list[k].isClosed()]])
            open_short_rings = [idx for idx in short_rings if
                                not ring_list[idx].isClosed()]
            num_short_and_closed = len(short_rings)-len(open_short_rings)
            if num_short_and_closed:
                sug_tol = (opt.tol_isNear * shortest_ring_length /
                           opt.appropriate_ring_length_minimum)
                warn("{} inappropriately short closed rings detected (and not "
                     "removed as "
                     "dont_remove_closed_inappropriately_short_rings = True). "
                     " You should probably decrease tol_isNear to something "
                     "less than {} and restart this file."
                     "".format(num_short_and_closed, sug_tol))
            short_rings = open_short_rings

        if opt.remove_inappropriately_short_rings:
            opt.basic_output_on.dprint("\nRemoving inappropriately short "
                                       "rings...", 'nr')
            ring_list = [ring for idx, ring in enumerate(ring_list)
                         if idx not in short_rings]
            opt.basic_output_on.dprint("Done (%s inappropriately short rings "
                                       "removed)." % (tmp_len - len(ring_list)))
        else:
            warn("{} inappropriately short rings were found, but "
                 "remove_inappropriately_short_rings is set to False."
                 "".format(len(ring_list)))
        print("")

    # Remove very short segments from rings
    def _remove_seg(path, _seg_idx, _newjoint):
        _new_path = [x for x in path]
        pathisclosed = path[-1].end == path[0].start

        # stretch next segment
        if _seg_idx != len(path) - 1 or pathisclosed:
            old_bpoints = _new_path[(_seg_idx + 1) % len(path)].bpoints()
            new_bpoints = (_newjoint,) + old_bpoints[1:]
            _new_path[(_seg_idx + 1) % len(path)] = bezier_segment(*new_bpoints)

        # stretch previous segment
        if _seg_idx != 0 or pathisclosed:
            old_bpoints = _new_path[(_seg_idx - 1) % len(path)].bpoints()
            new_bpoints = old_bpoints[:-1] + (_newjoint,)
            _new_path[(_seg_idx - 1) % len(path)] = bezier_segment(*new_bpoints)

        # delete the path to be removed
        del _new_path[_seg_idx]
        return _new_path

    if opt.min_relative_segment_length > 0:
        for r_idx, r in enumerate(ring_list):
            min_seg_length = r.path.length() * opt.min_relative_segment_length
            new_path = [s for s in r.path]
            its = 0
            flag = False
            while its < len(r.path):
                its += 1
                for seg_idx, seg in enumerate(new_path):
                    if seg.length() < min_seg_length:
                        flag = True
                        if seg == new_path[-1] and not r.path.isclosed():
                            newjoint = seg.end
                        elif seg == new_path[0].start and not r.path.isclosed():
                            newjoint = seg.start
                        else:
                            newjoint = seg.point(0.5)
                        new_path = _remove_seg(new_path, seg_idx, newjoint)
                        break
                else:
                    break
            if flag:
                ring_list[r_idx].path = Path(*new_path)

    # Close approximately closed rings
    for r in ring_list:
        r.fixClosure()

    # Palette check
    from svg2rings import palette_check
    ring_list = palette_check(ring_list)

    # Check for and fix inconsistencies in closedness of rings
    from svg2rings import closedness_consistency_check
    ring_list = closedness_consistency_check(ring_list)

    # Remove self-intersections in open rings
    if opt.remove_self_intersections:
        rsi_start_time = current_time()
        fixable_count = 0
        print("Checking for self-intersections...")
        bad_rings = []
        for r_idx, r in enumerate(ring_list):
            if r.path.end == r.path.start:
                continue
            first_half = r.path.cropped(0, 0.4)
            second_half = r.path.cropped(0.6, 1)
            middle_peice = r.path.cropped(0.4, 0.6)
            inters = first_half.intersect(second_half)
            if inters:
                if len(inters) > 1:
                    Ts = [info1[0] for info1, info2 in inters]
                    bad_rings.append((r_idx, Ts))
                    continue
                else:
                    fixable_count += 1
                T1, seg1, t1 = inters[0][0]
                T2, seg2, t2 = inters[0][1]

                T1 = crop_to_unit_interval(T1)
                T2 = crop_to_unit_interval(T2)

                if not opt.force_remove_self_intersections:
                    print("Self-intersection detected!")
                    greenpart = first_half.cropped(0, T1)
                    redpart = second_half.cropped(T2, 1)

                new_path = [seg for seg in first_half.cropped(T1, 1)]
                new_path += [seg for seg in middle_peice]
                new_path += [seg for seg in second_half.cropped(0, T2)]
                new_path = Path(*new_path)

                if opt.force_remove_self_intersections:
                    dec = True
                else:
                    print("Should I remove the red and green sections?")
                    disvg([greenpart, new_path, redpart],
                          ['green', 'blue', 'red'],
                          nodes=[seg1.point(t1)])
                    dec = inputyn()

                if dec:
                    r.path = new_path
                    print("Path cropped.")
                else:
                    print("OK... I hope things work out for you.")
        if bad_rings:
            paths = [r.path for r in ring_list]
            colors = [r.color for r in ring_list]
            center_line = Line(center-1, center+1)
            nodes = []
            for r_idx, Ts in bad_rings:
                for T in Ts:
                    nodes.append(ring_list[r_idx].path.point(T))
                colors[r_idx] = opt.colordict['safe2']
            node_colors = [opt.colordict['safe1']] * len(nodes)

            tmp = svgname + "_SelfIntersections.svg"
            fixed_svg_filename = os.path.join(opt.output_directory, tmp)
            disvg(paths + [center_line],
                  colors + [opt.colordict['center']],
                  nodes=nodes,
                  node_colors=node_colors,
                  filename=fixed_svg_filename)
            tmp_mes = (
                "Some rings contained multiple self-intersections, you better "
                "take a look.  They must be fixed manually (in Inkscape or "
                "Adobe Illustrator). An svg has been output highlighting the "
                "rings which must be fixed manually (and the points where the "
                "self-intersections occur).  Fix the highlighted rings and "
                "replace your old svg with the fixed one (the colors/circles "
                "used to highlight the intersections will be fixed/removed "
                "automatically).\n Output svg saved to:\n"
                "{}".format(fixed_svg_filename))
            raise Exception(tmp_mes)

        et = format_time(current_time()-rsi_start_time)
        print("Done fixing self-intersections ({} detected in {})."
              "".format(fixable_count, et))

    # Check that all rings are smooth (search for kinks and round them)
    if opt.smooth_rings:
        print("Smoothing paths...")
        bad_rings = []
        for r_idx, r in enumerate(ring_list):
            args = (r.path, opt.maxjointsize, opt.tightness, True)
            r.path = smoothed_path(*args)
            still_kinky_list = kinks(r.path)
            if still_kinky_list:
                bad_rings.append((r_idx, still_kinky_list))

        # If unremovable kinks exist, tell user to remove them manually
        if opt.ignore_unremovable_kinks or not bad_rings:
            opt.rings_may_contain_unremoved_kinks = False
        else:
            paths = [r.path for r in ring_list]
            colors = [r.color for r in ring_list]
            center_line = Line(center-1, center+1)
            nodes = []
            for r_idx, kink_indices in bad_rings:
                for idx in kink_indices:
                    kink = ring_list[r_idx].path[idx].start
                    nodes.append(kink)
                colors[r_idx] = opt.colordict['safe2']
            node_colors = [opt.colordict['safe1']] * len(nodes)

            fixed_svg_filename = os.path.join(
                opt.output_directory, svgname + "_kinks.svg")
            disvg(paths + [center_line],
                  colors + [opt.colordict['center']],
                  nodes=nodes,
                  node_colors=node_colors,
                  filename=fixed_svg_filename)
            raise Exception("Some rings contained kinks which could not be "
                            "removed automatically.  "
                            "They must be fixed manually (in inkscape or "
                            "adobe illustrator). An svg has been output "
                            "highlighting the rings which must be fixed "
                            "manually (and the points where the "
                            "kinks occur).  Fix the highlighted "
                            "rings and replace your old svg with the fixed "
                            "one (the colors/circles used to highlight the "
                            "kinks will be fixed/removed automatically).\n"
                            "Output svg saved to:\n"
                            "%s" % fixed_svg_filename)
        print("Done smoothing paths.")

    # Check for overlapping ends in open rings
    if opt.check4overlappingends:
        print("Checking for overlapping ends (that do not intersect)...")
        bad_rings = []
        for r_idx, r in enumerate(ring_list):
            if r.path.isclosed():
                continue
            startpt = r.path.start
            endpt = r.path.end
            path_wo_start = r.path.cropped(.1, 1)
            path_wo_end = r.path.cropped(0, .9)
            start_is_outwards = isPointOutwardOfPath(startpt, path_wo_start)
            end_is_outwards = isPointOutwardOfPath(endpt, path_wo_end)
            if start_is_outwards:
                bad_rings.append((r_idx, 0, start_is_outwards))
            if end_is_outwards:
                bad_rings.append((r_idx, 1, end_is_outwards))

        if bad_rings:
            paths = [r.path for r in ring_list]
            colors = [r.color for r in ring_list]
            center_line = Line(center-1, center+1)
            for r_idx, endbin, segts in bad_rings:
                colors[r_idx] = opt.colordict['safe2']

            # indicator lines
            indicator_lines = []
            for r_idx, endbin, segts in bad_rings:
                bad_path = ring_list[r_idx].path
                endpt = bad_path.point(endbin)
                for bad_seg_idx, bad_t in segts:
                    bad_pt = bad_path[bad_seg_idx].point(bad_t)
                    indicator_lines.append(Line(bad_pt, endpt))
            indicator_cols = [opt.colordict['safe1']] * len(indicator_lines)

            fixed_svg_filename = os.path.join(opt.output_directory,
                                              svgname + "_OverlappingEnds.svg")
            disvg(paths + [center_line] + indicator_lines,
                  colors + [opt.colordict['center']] + indicator_cols,
                  filename=fixed_svg_filename)
            bad_ring_count = len(set(x[0] for x in bad_rings))
            tmp_mes = (
                "Detected {} rings with overlapping (but not intersecting) "
                "ends.  They must be fixed manually (e.g. in Inkscape or "
                "Adobe Illustrator).  An svg has been output highlighting the "
                "rings which must be fixed manually.  Fix the highlighted "
                "rings, remove the,indicator lines added, and replace your "
                "old svg with the fixed one (the colors used to highlight the "
                "intersections will be fixed automatically).\nIf the "
                "indicator lines do not appear to be normal to the ring, this "
                "is possibly caused by a very short path segment.  In this "
                "case, you may want to try increasing "
                "min_relative_segment_length in options and running again.\n"
                "Output svg saved to:\n"
                "{}".format(bad_ring_count, fixed_svg_filename))
            raise Exception(tmp_mes)
        print("Done checking for overlapping ends.")

    # Trim paths with high curvature (i.e. curly) ends
    if opt.remove_curly_ends:
        print("Trimming high curvature ends...")
        for ring in ring_list:
            if ring.isClosed():
                continue

            # 90 degree turn in distance of opt.tol_isNear
            tol_curvature = 2**.5 / opt.tol_isNear  # #### Tolerance

            # Find any points within tol_isNear of start and end that have
            # curvature equal to tol_curvature, later we'll crop them off
            from svgpathtools import real, imag
            from svgpathtools.polytools import polyroots01

            def icurvature(segment, kappa):
                """returns a list of t-values such that 0 <= t<= 1 and
                seg.curvature(t) = kappa."""
                z = segment.poly()
                x, y = real(z), imag(z)
                dx, dy = x.deriv(), y.deriv()
                ddx, ddy = dx.deriv(), dy.deriv()

                p = kappa**2*(dx**2 + dy**2)**3 - (dx*ddy - ddx*dy)**2
                return polyroots01(p)

            # For first segment
            startseg = ring.path[0]
            ts = icurvature(startseg, tol_curvature)
            ts = [t for t in ts if startseg.length(t1=t) < opt.tol_isNear]
            if ts:
                T0 = ring.path.t2T(0, max(ts))
            else:
                T0 = 0

            # For last segment
            endseg = ring.path[-1]
            ts = icurvature(endseg, tol_curvature)
            ts = [t for t in ts if endseg.length(t0=t) < opt.tol_isNear]
            if ts:
                T1 = ring.path.t2T(-1, min(ts))
            else:
                T1 = 1

            # crop (if necessary)
            if T0 != 0 or T1 != 1:
                ring.path = ring.path.cropped(T0, T1)

        print("Done trimming.")

    # Check that there are no rings end outside the boundary ring (note
    # intersection removal in next step makes this sufficient)
    print("Checking for rings outside boundary ring...")
    boundary_ring = max([r for r in ring_list if r.isClosed()],
                        key=lambda rgn: rgn.maxR)
    outside_mark_indices = []
    for idx, r in enumerate(ring_list):
        if r is not boundary_ring:
            pt_outside_bdry = center + 2*boundary_ring.maxR
            if not ptInsideClosedPath(r.path[0].start,
                                      pt_outside_bdry,
                                      boundary_ring.path):
                outside_mark_indices.append(idx)
    if outside_mark_indices:
        ring_list = [r for i, r in enumerate(ring_list)
                     if i not in outside_mark_indices]
        warn("%s paths were found outside the boundary path and will be "
             "ignored." % len(outside_mark_indices))
    print("Done removing rings outside of boundary ring.")

    # Remove intersections (between distinct rings)
    if opt.rings_may_contain_intersections:
        print("Removing intersections (between distinct rings)...")
        from noIntersections4rings import remove_intersections_from_rings
        opt.basic_output_on.dprint(
            "Now attempting to find and remove all intersections from "
            "rings (this will take a long time)...")
        intersection_removal_start_time = current_time()

        ring_list, intersection_count, overlappingClosedRingPairs = \
            remove_intersections_from_rings(ring_list)

        if not overlappingClosedRingPairs:
            tot_ov_time = format_time(current_time() - intersection_removal_start_time)
            opt.basic_output_on.dprint(
                "Done (in just %s). Found and removed %s intersections."
                "" % (tot_ov_time, intersection_count))
        else:
            # fixed_paths = [parse_path(r.string) for r in ring_list]
            fixed_paths = [r.path for r in ring_list]
            fixed_colors = [r.color for r in ring_list]
            center_line = Line(center-1, center+1)
            nodes = []
            for i, j in overlappingClosedRingPairs:
                fixed_colors[i] = opt.colordict['safe1']
                fixed_colors[j] = opt.colordict['safe2']
                inters = pathXpathIntersections(
                    ring_list[i].path, ring_list[j].path)
                nodes += [inter[0].point(inter[2]) for inter in inters]

            fixed_svg_filename = os.path.join(
                opt.output_directory, svgname + "_ClosedRingsOverlap.svg")
            disvg(fixed_paths + [center_line],
                  fixed_colors + [opt.colordict['center']],
                  nodes=nodes,
                  filename=fixed_svg_filename)
            raise Exception("Found %s pair(s) over overlapping closed rings.  "
                            "They must be fixed manually (in inkscape or "
                            "adobe illustrator). An svg has been output "
                            "highlighting the rings which must be separated "
                            "manually (and the points where they intersect).  "
                            "Fix the highlighted rings and replace your old "
                            "svg with the fixed one (the colors/circles used "
                            "to highlight the intersections will be "
                            "fixed/removed automatically).\n"
                            "Output svg saved to:\n"
                            "%s" % (len(overlappingClosedRingPairs),
                                    fixed_svg_filename))

    # Output a fixed SVG that is (hopefully) how this SVG would be if humans
    # were perfect
    if opt.create_fixed_svg:
        opt.basic_output_on.dprint("Now creating a fixed svg file...", 'nr')
        fixed_paths = [r.path for r in ring_list]
        fixed_colors = [r.color for r in ring_list]
        center_line = Line(center - 1, center + 1)

        fixed_svg_filename = os.path.join(opt.output_directory,
                                          svgname + "_fixed.svg")
        wsvg(fixed_paths + [center_line],
             fixed_colors + [opt.colordict['center']],
             filename=fixed_svg_filename)
        opt.basic_output_on.dprint("Done.  SVG file saved to:\n"
                                   "%s" % fixed_svg_filename)
