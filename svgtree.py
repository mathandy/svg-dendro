#External Module Dependencies
from __future__ import division
from os import listdir, makedirs as os_makedirs, path as os_path
from ntpath import basename as nt_path_basename
import cPickle as pickle
from time import time as current_time
from warnings import warn
from svgpathtools import (parse_path, Path, Line, disvg, wsvg, kinks,
                          smoothed_path, bezier_segment)
from andysmod import format_time, Radius, inputyn, ask_user
import numpy as np

#Internal Module Dependencies
from misc4rings import (displaySVGPaths_transects, plotUnraveledRings,
                        pathXpathIntersections)
from andysSVGpathTools import ptInsideClosedPath
from transects4rings import isPointOutwardOfPath
from svg2rings import svg2rings
import options4rings as opt


######################################################################################
###Check for human errors and create more perfect SVG ################################
######################################################################################

def fix_svg(ring_list, center):

    #Discard inappropriately short rings
    from options4rings import appropriate_ring_length_minimum
    opt.basic_output_on.dprint("\nChecking for inappropriately short rings...",'nr')
    tmp_len = len(ring_list)
    short_rings = [idx for idx,ring in enumerate(ring_list) if ring.path.length()<appropriate_ring_length_minimum]
    opt.basic_output_on.dprint("Done (%s inappropriately short rings found)."%len(short_rings))
    if short_rings:
        if opt.create_svg_highlighting_inappropriately_short_rings:
            opt.basic_output_on.dprint("\nCreating svg highlighting inappropriately short rings...",'nr')
            paths = [parse_path(r.string) for r in ring_list]
            colors = [r.color for r in ring_list]
            nodes = [ring_list[idx].path.point(0.5) for idx in short_rings]
            center_line = [Line(center-1,center+1)]
            tmp = svgfile[0:len(svgfile)-4] + "_short-rings.svg"
            shortrings_svg_filename = os_path.join(opt.outputFolder, tmp)
            disvg(paths + [center_line], colors + [opt.colordict['center']],
                  nodes=nodes, filename=shortrings_svg_filename)
            args = appropriate_ring_length_minimum, shortrings_svg_filename
            mes = ("Done.  SVG created highlighting short rings by placing a node at "
                   "each short ring's midpoint.  Note: since these rings are all under "
                   "{} pixels in length, they may be hard to see and may even be "
                   "completely covered by the node.SVG file saved to:\n{}").format(*args)
            opt.basic_output_on.dprint(mes)
        if opt.dont_remove_closed_inappropriately_short_rings:
            shortest_ring_length = min([r.path.length() for r in [ring_list[idx] for idx in short_rings if ring_list[idx].isClosed()]])
            open_short_rings = [idx for idx in short_rings if not ring_list[idx].isClosed()]
            num_short_and_closed = len(short_rings)-len(open_short_rings)
            if num_short_and_closed:
                sug_tol = opt.tol_isNear * shortest_ring_length / opt.appropriate_ring_length_minimum
                warn("%s inappropriately short closed rings detected (and not "
                     "removed as "
                     "dont_remove_closed_inappropriately_short_rings = True). "
                     " You should probably decrease tol_isNear to something "
                     "less than %s and restart this file." % (num_short_and_closed,
                                                              sug_tol))
            short_rings = open_short_rings
        if opt.remove_inappropriately_short_rings:
            opt.basic_output_on.dprint("\nRemoving inappropriately short rings...",'nr')
            ring_list = [ring for idx,ring in enumerate(ring_list) if idx not in short_rings]
            opt.basic_output_on.dprint("Done (%s inappropriately short rings removed)."%(tmp_len-len(ring_list)))
        else:
            warn("%s inappropriately short rings were found, but "
                 "remove_inappropriately_short_rings is set to False." % len(ring_list))
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

        #delete the path to be removed
        del _new_path[_seg_idx]
        return _new_path



    if opt.min_relative_segment_length:
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
        print "Checking for self-intersections..."
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
                if not opt.force_remove_self_intersections:
                    print "Self-intersection detected!"
                    greenpart = first_half.cropped(0, T1)
                    redpart = second_half.cropped(T2, 1)

                new_path = [seg for seg in first_half.cropped(T1, 1)]
                new_path += [seg for seg in middle_peice]
                new_path += [seg for seg in second_half.cropped(0, T2)]
                new_path = Path(*new_path)

                if opt.force_remove_self_intersections:
                    dec = True
                else:
                    print "Should I remove the red and green sections?"
                    disvg([greenpart, new_path, redpart],
                          ['green', 'blue', 'red'],
                          nodes=[seg1.point(t1)])
                    dec = inputyn()

                if dec:
                    r.path = new_path
                    print "Path cropped."
                else:
                    print "OK... I hope things work out for you."
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

            tmp = svgfile[0:len(svgfile)-4] + "_SelfIntersections.svg"
            fixed_svg_filename = os_path.join(opt.outputFolder, tmp)
            disvg(paths + [center_line],
                  colors + [opt.colordict['center']],
                  nodes=nodes,
                  node_colors=node_colors,
                  filename=fixed_svg_filename)
            raise Exception("Some rings contained multiple self-intersections,"
                            " you better take a look.  "
                            "They must be fixed manually (in inkscape or "
                            "adobe illustrator). An svg has been output "
                            "highlighting the rings which must be fixed "
                            "manually (and the points where the "
                            "self-intersections occur).  Fix the highlighted "
                            "rings and replace your old svg with the fixed "
                            "one (the colors/circles used to highlight the "
                            "intersections will be fixed/removed "
                            "automatically).\n"
                            "Output svg saved to:\n"
                            "%s" % fixed_svg_filename)

        et = format_time(current_time()-rsi_start_time)
        print "Done fixing self-intersections (%s detected in %s)." % (fixable_count, et)


    # Check that all rings are smooth (search for kinks and round them)
    if opt.smooth_rings:
        print "Smoothing paths..."
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

            tmp = svgfile[0:len(svgfile)-4] + "_kinks.svg"
            fixed_svg_filename = os_path.join(opt.outputFolder, tmp)
            disvg(paths + [center_line],
                  colors + [opt.colordict['center']],
                  nodes=nodes,
                  node_colors = node_colors,
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
        print "Done smoothing paths."

    # Check for overlapping ends in open rings
    if opt.check4overlappingends:
        print "Checking for overlapping ends (that do not intersect)..."
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
                # bla1 = isPointOutwardOfPath(startpt, path_wo_start)  # debug line
            if end_is_outwards:
                bad_rings.append((r_idx, 1, end_is_outwards))
                # bla2 = isPointOutwardOfPath(endpt, path_wo_end)  # debug line

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

            tmp = svgfile[0:len(svgfile)-4] + "_OverlappingEnds.svg"
            fixed_svg_filename = os_path.join(opt.outputFolder, tmp)
            disvg(paths + [center_line] + indicator_lines,
                  colors + [opt.colordict['center']] + indicator_cols,
                  filename=fixed_svg_filename)
            bad_ring_count = len(set(x[0] for x in bad_rings))
            raise Exception("Detected %s rings with overlapping (but not "
                            "intersecting) ends.  "
                            "They must be fixed manually (in inkscape or "
                            "adobe illustrator). An svg has been output "
                            "highlighting the rings which must be fixed "
                            "manually.  Fix the highlighted rings, remove the,"
                            "indicator lines added, and replace your old svg "
                            "with the fixed one (the colors used to highlight "
                            "the intersections will be fixed automatically).\n"
                            "If the indicator lines do not appear to be "
                            "normal to the ring, this is possibly caused by a "
                            "very short path segment.  In this case, you may"
                            "want to try increasing "
                            "min_relative_segment_length in options and "
                            "running again.\n"
                            "Output svg saved to:\n"
                            "%s" % (bad_ring_count, fixed_svg_filename))
        print "Done checking for overlapping ends."

    # Trim paths with high curvature (i.e. curly) ends
    if opt.remove_curly_ends:
        def kappa(seg, t):
            try:
                return seg.curvature(t)
            except ValueError:
                return 0

        print "Trimming high curvature ends..."
        for ring in ring_list:
            if ring.isClosed():
                continue

            # Find estimate of max curvature of inner part of ring
            segCurvatures = []
            for seg in ring.path[1:-1]:
                segCurvatures.append(max(kappa(seg, t)
                                     for t in np.linspace(0, 1, 10)))
            for seg in [ring.path[0], ring.path[-1]]:
                segCurvatures.append(max(kappa(seg, t)
                                     for t in np.linspace(.2, .8, 5) ))
#                    aveCurvature = aveCurvature/(len(ring.path)*10)

            tol_curvature = 20 * max(segCurvatures)  #####Tolerance

            # Initialize variables used to remember where to crop
            t0, t1 = 0, 1

            mes2 = ("  Set manually_curly_end=True to fix this ring.")

            # check first segment in each ring
            mes1 = ("A curl has been detected that lasts for most or all the "
                "first segment!")
            startseg = ring.path[0]
            for k in range(0, 50):
                t = 0.5 - k/100
                if kappa(startseg, t) > tol_curvature:
                    if k < 10:
                        if opt.manually_curly_end:
                            print mes1
                            print "Should I remove this red segment?"
                            if len(ring.path) != 1:
                                greenpath = Path(*ring.path[1:])
                                redpath = startseg
                            else:
                                greenpath = startseg.cropped(t, 1)
                                redpath = startseg.cropped(0, t)
                            disvg([greenpath, redpath], ['green', 'red'],
                                  nodes=[redpath.start, redpath.end])
                            if inputyn():
                                t0 = t
                        elif opt.ignore_long_curls:
                            warn(mes1 + mes2 + "  Continuing... and hoping "
                                "this doesn't cause any problems.")
                        else:
                            raise Exception(mes1 + mes2)
                    else:
                        t0 = t
            T0 = ring.path.t2T(0, t0)

            # check first segment in each ring
            mes1 = ("A curl has been detected that lasts for most or all "
                        "the last segment!")
            endseg = ring.path[-1]
            for k in range(0, 50):
                t = k/100
                try:
                    endseg_curvature = endseg.curvature(t)
                    flag = False
                except ValueError:
                    flag = True
                if flag or endseg_curvature > tol_curvature:
                    if k < 10:
                        if opt.manually_curly_end:
                            print (mes1)
                            print "Should I remove this red segment?"
                            if len(ring.path) != 1:
                                greenpath = Path(*ring.path[:-1])
                                redpath = endseg
                            else:
                                greenpath = endseg.cropped(0, t)
                                redpath = endseg.cropped(t, 1)
                            disvg([greenpath, redpath], ['green', 'red'],
                                  nodes=[redpath.start, redpath.end])
                            if inputyn():
                                t1 = t
                        elif opt.ignore_long_curls:
                            warn(mes1 + mes2 + "Continuing... and hoping "
                                 "this doesn't cause any problems.")
                        else:
                            raise Exception(mes1 + mes2)
                    else:
                        t1 = t
            T1 = ring.path.t2T(len(ring.path) - 1, t1)
            if T0 != 0 or T1 != 1:
                ring.path = ring.path.cropped(T0, T1)
        print "Done trimming."


    # Check that there are no rings end outside the boundary ring (note
    # intersection removal in next step makes this sufficient)
    print "Checking for rings outside boundary ring..."
    boundary_ring = max([r for r in ring_list if r.isClosed()],
                        key=lambda rgn: rgn.maxR)
    outside_mark_indices = []
    for idx,r in enumerate(ring_list):
        if r is not boundary_ring:
            pt_outside_bdry = center + 2*boundary_ring.maxR
            if not ptInsideClosedPath(r.path[0].start,
                                      pt_outside_bdry,
                                      boundary_ring.path):
                outside_mark_indices.append(idx)
    if outside_mark_indices:
        ring_list = [r for i,r in enumerate(ring_list)
                     if i not in outside_mark_indices]
        warn("%s paths were found outside the boundary path and will be "
             "ignored." % len(outside_mark_indices))
    print "Done removing rings outside of boundary ring."


    # Remove intersections (between distinct rings)
    if opt.rings_may_contain_intersections:
        print "Removing intersections (between distinct rings)..."
        from noIntersections4rings import remove_intersections_from_rings
        opt.basic_output_on.dprint("Now attempting to find and remove all "
                               "intersections from rings (this will take a "
                               "long time)...")
        intersection_removal_start_time = current_time()
        ring_list, intersection_count, overlappingClosedRingPairs = remove_intersections_from_rings(ring_list)
        if not overlappingClosedRingPairs:
            tot_ov_time = format_time(current_time() - intersection_removal_start_time)
            opt.basic_output_on.dprint("Done (in just %s). Found and removed %s "
                                   "intersections." % (tot_ov_time,
                                                       intersection_count))
        else:
            # fixed_paths = [parse_path(r.string) for r in ring_list]
            fixed_paths = [r.path for r in ring_list]
            fixed_colors = [r.color for r in ring_list]
            center_line = Line(center-1, center+1)
            nodes = []
            for i, j in overlappingClosedRingPairs:
                fixed_colors[i] = opt.colordict['safe1']
                fixed_colors[j] = opt.colordict['safe2']
                inters = pathXpathIntersections(ring_list[i].path,ring_list[j].path)
                nodes += [inter[0].point(inter[2]) for inter in inters]

            tmp = svgfile[0:len(svgfile)-4] + "_ClosedRingsOverlap.svg"
            fixed_svg_filename = os_path.join(opt.outputFolder, tmp)
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
    from options4rings import create_fixed_svg
    if create_fixed_svg:
        opt.basic_output_on.dprint("Now creating a fixed svg file...", 'nr')
        fixed_paths = [r.path for r in ring_list]
        fixed_colors = [r.color for r in ring_list]
        center_line = Line(center-1, center+1)
        fixed_svg_filename = os_path.join(opt.outputFolder, svgfile[0:len(svgfile)-4] + "_fixed.svg")
        wsvg(fixed_paths + [center_line],
              fixed_colors + [opt.colordict['center']],
              filename=fixed_svg_filename)
        opt.basic_output_on.dprint("Done.  SVG file saved to:\n"
                                   "%s" % fixed_svg_filename)

def svgtree():
    #determine if pickle file exists, if it does, load ring_list and center from it
    if opt.ignore_extant_pickle_file:
        pickle_file_exists = False
    else:
        pickle_file_exists = True
        try:
            (ring_list, center) = pickle.load(open(pickle_file, "rb"))
        except:
            pickle_file_exists = False

    #determine if sorted pickle file exists, if it does, load ring_list and center from it (instead of unsorted pickle)
    if opt.ignore_extant_sorted_pickle_file:
        sorted_pickle_file_exists = False
    else:
        sorted_pickle_file_exists = True
        try:
            (ring_list, center) = pickle.load(open(sorted_pickle_file, "rb"))
        except:
            sorted_pickle_file_exists = False

    # If pickle file doesn't exist, create one, and store ring_list and center in it
    if not (pickle_file_exists or sorted_pickle_file_exists): ## Load the dictionary back from the pickle file.
        center, ring_list = svg2rings(SVGfileLocation)
        opt.basic_output_on.dprint("Pickling ring_list... ",'nr')
        pickle.dump((ring_list,center), open(pickle_file, "wb"))
        opt.basic_output_on.dprint('pickling complete -> ' + pickle_file)
        opt.basic_output_on.dprint("Done.")
    rad = Radius(center)

######################################################################################
###Ad hoc fix to record svg names in rings ###########################################
######################################################################################
    for ring in ring_list:
        ring.svgname = svgfile[:-4]

    if not opt.assume_svg_is_fixed:
        fix_svg(ring_list, center)

###############################################################################
###Sort #######################################################################
###############################################################################
    #sort ring_list from innermost to outermost and record sort index
    if opt.sort_rings_on:
        if not sorted_pickle_file_exists:
            tmp_mes =("Attempting to sort ring_list.  This could take a minute "
                      "(or thirty)...")
            opt.basic_output_on.dprint(tmp_mes,'nr')
            #find sorting of ring_list
            from sorting4rings import sort_rings
#                    ring_sorting, psorting = sort_rings(ring_list, om_pickle_file)
            ring_sorting, sort_lvl_info = sort_rings(ring_list, om_pickle_file)
            opt.basic_output_on.dprint("Done sorting ring_list.")

            #record sort index
            for i, r_index in enumerate(ring_sorting):
                ring_list[r_index].sort_index = i
#                        ring_list[r_index].psort_index = ???

            # pickle "sorted" ring_list (not really sorted, but sort_index's
            # are recorded)
            opt.basic_output_on.dprint("Pickling sorted ring_list... ", 'nr')
            pickle.dump((ring_list, center), open(sorted_pickle_file, "wb"))
            opt.basic_output_on.dprint('pickling complete -> ' +
                                       sorted_pickle_file)

###############################################################################
###Generate and Output Transects ##############################################
###############################################################################

    #generate transects
    skipped_angle_indices = []
    opt.basic_output_on.dprint("Generating the %s requested transects..." % opt.N_transects, 'nr')
    if opt.N_transects > 0:
        if opt.use_ring_sort_4transects:
            if opt.generate_evenly_spaced_transects:
                from transects4rings import generate_inverse_transects
                from numpy import linspace
                def inv_arclength(curve, s):
                    return curve.ilength(s)
                angles = linspace(0, 1, opt.N_transects+1)[:-1]
                bdry_ring = max(ring_list, key=lambda r: r.maxR)
                bdry_length = bdry_ring.path.length()
                Tvals = [inv_arclength(bdry_ring.path, s*bdry_length) for s in angles]
                data, data_indices, skipped_angle_indices = generate_inverse_transects(ring_list, Tvals)
                num_suc = len(data)
                nums = (num_suc, opt.N_transects, opt.N_transects - num_suc)
                trmes = ("%s / %s evenly spaced transects successfully "
                         "generated (skipped %s)." % nums)
                opt.basic_output_on.dprint(trmes)
            else:
                from transects4rings import generate_sorted_transects
                data, data_indices, angles = generate_sorted_transects(ring_list, center, angles2use=opt.angles2use)
        else:
            from transects4rings import generate_unsorted_transects
            data, data_indices, angles = generate_unsorted_transects(ring_list, center)
        opt.basic_output_on.dprint("Done generating transects.")

        #show them (this creates an svg file in the root folder)
        if opt.create_SVG_picture_of_transects:
            svgname = svgfile[0:len(svgfile)-4]
            svg_trans = os_path.join(opt.outputFolder, svgname + "_transects.svg")
            displaySVGPaths_transects(ring_list, data, angles, skipped_angle_indices, fn=svg_trans)
            tmp_mes = "\nSVG showing transects generated saved to:\n" + svg_trans + "\n"
            opt.basic_output_on.dprint(tmp_mes)

        #Save results from transects
        from transects4rings import save_transect_data, save_transect_summary
        completed_angles = [x for idx, x in enumerate(angles)
                            if idx not in skipped_angle_indices]
        skipped_angles = [angles[idx] for idx in skipped_angle_indices]
        save_transect_data(outputFile_transects, ring_list, data, data_indices, completed_angles, skipped_angles)
        save_transect_summary(outputFile_transect_summary, ring_list, data, data_indices, completed_angles)

######################################################################################
###Compute Ring Areas ################################################################
######################################################################################
    if opt.find_areas:
        from area4rings import find_ring_areas
        sorted_ring_list = sorted(ring_list,key=lambda rg:rg.sort_index)
        find_ring_areas(sorted_ring_list, center, svgfile) #this also completes incomplete rings

######################################################################################
###Other (optional) stuff ############################################################
######################################################################################

    #Create SVG showing ring sorting
    if opt.create_SVG_showing_ring_sort:
        opt.basic_output_on.dprint("Attempting to create SVG showing ring sorting...",'nr')
        from misc4rings import displaySVGPaths_numbered
        tmp = svgfile[0:len(svgfile)-4] + "_sort_numbered"+".svg"
        svgname = os_path.join(opt.outputFolder_debug, tmp)
        displaySVGPaths_numbered([r.path for r in ring_list],svgname,[r.color for r in ring_list])
        opt.basic_output_on.dprint("Done.")



    #test complete ring sort after first sort round
    if opt.visual_test_of_all_ring_sort_on:
        from svg2rings import visual_test_of_ring_sort
        visual_test_of_ring_sort(ring_list)

    #plot all rings on a plot with x = theta and y = r (if opt.showUnraveledRingPlot is set to True)
    if opt.showUnraveledRingPlot:
        opt.basic_output_on.dprint("Creating unraveled ring plot... ",'nr')
        plotUnraveledRings(ring_list,center)
        opt.basic_output_on.dprint("Done.  (It should have opened automatically and now be visible.)")

###############################################################################
###Report Success/Failure of file #############################################
###############################################################################

    opt.basic_output_on.dprint("Success! Completed %s in %s."%(svgfile,format_time(current_time()-file_start_time)))
    opt.basic_output_on.dprint(":)"*50)
    opt.basic_output_on.dprint("<>"*50)
    opt.basic_output_on.dprint("<>"*50)
    opt.basic_output_on.dprint("\n\n")
    error_list.append((svgfile,"Completed Successfully."))



###############################################################################
# Check if output (sub)directories exist, create subdirectories if they don't
# exist
mes = ("\n\nThe outputFolder given in options does not exist.  To fix this "
      "change outputFolder in options, or create the folder:\n"
       "%s" % opt.outputFolder)
assert os_path.exists(opt.outputFolder), mes
if not os_path.exists(opt.outputFolder_pickles): #debug folder
    os_makedirs(opt.outputFolder_pickles)
if not os_path.exists(opt.outputFolder_debug): #pickle folder
    os_makedirs(opt.outputFolder_debug)
###############################################################################
###Batch run all SVG filed in input directory #################################
###############################################################################
error_list = []
num_of_files = len(listdir(opt.input_directory))
for svgfile_idx in range(num_of_files):

    # Allows option to start at at different file than first (mostly for
    # debugging purposes)
    shifted_index = (svgfile_idx + opt.start_at_file_number) % num_of_files
    svgfile = listdir(opt.input_directory)[shifted_index]

###############################################################################
###Load SVG, extract rings, pickle (or just load pickle if it exists) #########
###############################################################################
    if svgfile[len(svgfile)-3:len(svgfile)] == 'svg':
        file_start_time = current_time()
        try:
            print('-'*40+'\n'+'~'*20+'attempting %s'%svgfile+'\n'+'-'*40)
            SVGfileLocation = os_path.join(opt.input_directory, svgfile)
            extractedFileName = nt_path_basename(SVGfileLocation)[0:-4]

            # Name output csv and pickle Files
            tmp = 'DataFrom-' + extractedFileName + '.csv'
            outputFile = os_path.join(opt.outputFolder, tmp)
            tmp = 'TransectDataFrom-' + extractedFileName + '.csv'
            outputFile_transects = os_path.join(opt.outputFolder, tmp)
            tmp = 'TransectSummary-' + extractedFileName + '.csv'
            outputFile_transect_summary = os_path.join(opt.outputFolder, tmp)
            tmp = 'DataFrom-'+ extractedFileName +'_failed_rings.csv'
            outputFile_failed_rings = os_path.join(opt.outputFolder, tmp)
            tmp = extractedFileName + "-ring_list.p"
            pickle_file = os_path.join(opt.outputFolder_pickles, tmp)
            tmp = extractedFileName + "-sorted-ring_list.p"
            sorted_pickle_file = os_path.join(opt.outputFolder_pickles, tmp)
            tmp = extractedFileName + "-ordering_matrix.p"
            om_pickle_file = os_path.join(opt.outputFolder_pickles, tmp)

            # Analyze svgfile
            svgtree()
        except:
            print("-"*75)
            print("!"*25+svgfile+" did not finish successfully.")
            print("Reason:")

            #save error to error_list
            if opt.if_file_throws_error_skip_and_move_to_next_file:
                from traceback import format_exc
                from sys import stdout
                stdout.write(format_exc())
                error_list.append((svgfile,format_exc()))
                print("-"*75)
                print("VV"*50)
                print("VV"*50)
                print("\n\n")
                continue
            else:
                raise
    else:
        continue
print("")
error_log = os_path.join(opt.outputFolder, "error_list.txt")
print "error_list ouput to:\n%s"%error_log
with open(error_log, 'wt') as outf:
    for (svgname, err) in error_list:
        outf.write('#'*50+'\n')
        outf.write('### '+svgname+'\n')
        outf.write('#'*50+'\n')
        outf.write(err+'\n'*3)
print("All done.")