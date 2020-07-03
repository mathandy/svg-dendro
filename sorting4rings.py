# Internal Dependencies
from andysmod import (format_time, topo_sorted, createOrderingMatrix,
                      flattenList)
from misc4rings import (normalLineAtT_toInner_intersects_withOuter,
                        pathXpathIntersections, display2rings4user, dis)
from options4rings import (basic_output_on, use_alternative_sorting_method,
                           alt_sort_N, percentage_for_disagreement,
                           look_for_user_sort_input)
import options4rings as opt

# External Dependencies
from svgpathtools import Path, Line, path_encloses_pt, disvg, wsvg
from time import sleep, time as current_time
from itertools import combinations
from numpy import NaN, isnan, where, transpose
from operator import itemgetter
import _pickle as pickle
import os


def ring1_isbelow_ring2_numHits(ring1, ring2, n_test_lines, debug_name=''):
    """Computes the number (out of n_test_lines) of the checked lines
    from ring1 to the center that intersect  with ring2
    """
    center = ring1.center
    count_hits = 0
    tran_colors = []  # for debug mode
    tran_lines = []  # for debug mode
    for i in range(n_test_lines):
        innerT = i/(n_test_lines - 1)
        nlin, seg_out, t_out = normalLineAtT_toInner_intersects_withOuter(
            innerT, ring1.path, ring2.path, center, 'debug')

        if debug_name != '':  # output an SVG with the lines used
            tran_lines.append(nlin)
            tran_colors.append('black' if seg_out != False else 'purple')

        if seg_out != False:
            count_hits += 1

    if debug_name != '':
        dis([ring1.path, ring2.path],
            ['green', 'red'],
            lines=tran_lines,
            line_colors=tran_colors,
            filename=debug_name)
    return count_hits


def ring1_isknowntobebelow_ring2(ring1,ring2):
    if ring2 in ring1.isBelow or ring1 in ring2.isAbove:
        return True
    if ring1.isBelow.intersection(ring2.isAbove) != set([]):
        return True
    return False


def closedRing_cmp_ring_partial(cring,ring):
    if ring.minR < cring.minR:
        return 1 #cring is above ring
    if ring.maxR > cring.maxR:
        return -1 #cring is below ring
    return 0


def record_ring1_isbelow_ring2(ring1,ring2):
    ring1.isBelow.add(ring2)
    ring2.isAbove.add(ring1)
    ring1.isBelow = ring1.isBelow.union(ring2.isBelow)
    ring2.isAbove = ring2.isAbove.union(ring1.isAbove)
    return #nothing


def ring1_isabove_ring2_forCertain_cmp(ring1,ring2,sort_round=0,solo_round=False):
    if not solo_round or sort_round=='cc' or sort_round == 1:
        if ring1.isClosed() and ring2.isClosed():
            if ring1.minR < ring2.minR:
                return -1
            elif ring1.minR > ring2.minR:
                return 1
            else:
                raise Exception("This should never happen.  There must be a dupicate closed ring in ring_list.")
    if not solo_round or sort_round=='known' or sort_round == 1:
        if ring1_isknowntobebelow_ring2(ring1,ring2):
            return -1
        if ring1_isknowntobebelow_ring2(ring2,ring1):
            return 1
    #one closed ring (and no intersections)
    if not solo_round or sort_round=='one_closed' or sort_round == 1:
        if ring1.isClosed():
            res = closedRing_cmp_ring_partial(ring1,ring2)
            if res !=0 and pathXpathIntersections(ring1.path,ring2.path)==[]:
                if res == 1:
                    return 1
                if res == -1:
                    return -1
        elif ring2.isClosed():
            res = closedRing_cmp_ring_partial(ring2,ring1)
            if res !=0 and pathXpathIntersections(ring1.path,ring2.path)==[]:
                if res == 1:
                    return -1
                if res == -1:
                    return 1


def ask_user_to_sort(i,j,ring_list,make_svg=True,ask_later=True):
    #returns 1 if ring_list[i] is the more inner ring, -1 if ring_list[j] is and 0 if they are incomparable (or equal)
#    if i>j:
#        return -1*ask_user_to_sort(j,i,ring_list) #prevents asking user about same set of rings twice
    if ask_later: #save an svg for "interactive sorting" and return NaN
        from options4rings import output_directory
        from os import path as os_path
        save_loc = os_path.join(output_directory,'interactive_sorting',
                                ring_list[i].svgname,'cmp_%s-%s.svg'%(i,j))
        disvg([ring_list[i].path,ring_list[j].path],['green','red'],
              nodes=[ring_list[i].center],filename=save_loc)
        return NaN
    else:
        if make_svg:
            display2rings4user(i,j,ring_list)

        s = "Enter 1 if: Green is a more inner ring than red.\n"
        s+= "Enter -1 if: the opposite is true\n"
        s+= "Enter 0 if: neither is more inner according to tracing.\n"
        s+= "Enter im if this is an impossible case that must be fixed by hand."
        s+= "Enter g to output an svg with only the center, boundary, and green ring\n"
        s+= "Enter r to output an svg with only the center, boundary, and red ring\n"
        s+= "Enter b to output an svg with only the center and the red and green rings.\n"
        s+= "Enter db to output svgs showing the normal lines used to test the comparison.\n"
        s+= "Enter rb to output a runnable svg with only the center, boundary, and both the red and green rings.\n"
        s+= "Enter q to terminate program.\n"
        s+= "Your answer: "
        response = input(s)
        if response=='1':
            return 1
        elif response== '-1':
            return -1
        elif response== '0':
            return 0
        elif response in ['g','r','b','rb']:
            display2rings4user(i,j,ring_list,mode=response)
            return ask_user_to_sort(i,j,ring_list,make_svg=False,ask_later=ask_later)
        elif response=='db':
            ring1_isbelow_ring2_numHits(ring_list[i],ring_list[j],alt_sort_N,debug_name='db12.svg')
            ring1_isbelow_ring2_numHits(ring_list[j],ring_list[i],alt_sort_N,debug_name='db21.svg')
            return ask_user_to_sort(i,j,ring_list,make_svg=False,ask_later=ask_later)
        elif response=='q':
            raise Exception("User-forced termination of program.")
        else:
            return ask_user_to_sort(i,j,ring_list)


#def interp_ring_cmp(sgn):
#    if sgn==1:
#        return "%s (green is above red)"%sgn
#    if sgn==-1:
#        return "%s (red is above green)"%sgn
#    if sgn==0:
#        return "%s (uncertain)"%sgn
#    return "%s (no result returned)"%sgn
#def interp_ring_closure(isClosed):
#    if isClosed:
#        return 'Closed'
#    else:
#        return 'Open'


def postsort_ring1_isoutside_ring2_cmp(ring1,ring2):
    d = ring1.sort_index - ring2.sort_index
    return d/abs(d)
    
    
def ring1_isoutside_ring2_cmp_alt(ringlist, ring1_index, ring2_index,
                                  N_lines2use=opt.alt_sort_N,
                                  increase_N_if_zero=True):#####TOL
    """Returns 1 if true, -1 if false and 0 if equal"""
    ring1 = ringlist[ring1_index]
    ring2 = ringlist[ring2_index]
    if ring1.path == ring2.path:
        return 0
    countHits12 = ring1_isbelow_ring2_numHits(ring1, ring2, N_lines2use)
    countHits21 = ring1_isbelow_ring2_numHits(ring2, ring1, N_lines2use)
    if countHits12 == 0 or countHits21 == 0:
        if countHits12 > 0:
            return -1
        elif countHits21 > 0:
            return 1
        elif increase_N_if_zero:
            N_upped = N_lines2use * max(len(ring1.path), len(ring2.path))
            improved_res = ring1_isoutside_ring2_cmp_alt(
                ringlist, ring1_index, ring2_index, N_lines2use=N_upped,
                increase_N_if_zero=False)
            if improved_res != 0:
                return improved_res
            elif ring1.isClosed() or ring2.isClosed():
                if opt.manually_fix_sorting:
                    return ask_user_to_sort(
                        ring1_index, ring2_index, ringlist, make_svg=True)
                else:
                    raise Exception(
                        "Problem sorting rings... set "
                        "'manually_fix_sorting=True' in options4rings.py "
                        "to fix manually."
                    )
            else:
                return 0
        else:
            return 0

    # neither of the counts were zero
    ratio21over12 = countHits21/countHits12
    try:
        upper_bound = 1.0/percentage_for_disagreement
    except ZeroDivisionError:
        from numpy import Inf
        upper_bound = Inf

    if percentage_for_disagreement < ratio21over12< upper_bound:
        # still not sure, so use more lines
        N_upped = N_lines2use * max(len(ring1.path), len(ring2.path))
        countHits12 = ring1_isbelow_ring2_numHits(ring1, ring2, N_upped)
        countHits21 = ring1_isbelow_ring2_numHits(ring2, ring1, N_upped)
        ratio21over12 = countHits21/countHits12

        if percentage_for_disagreement < ratio21over12 < upper_bound:
            # still not sure, ask user, if allowed
            if opt.manually_fix_sorting:
                return ask_user_to_sort(
                    ring1_index, ring2_index, ringlist, make_svg=True)
            else:
                raise Exception(
                    "Problem sorting rings... set "
                    "'manually_fix_sorting=True' in options4rings.py to "
                    "fix manually."
                )
    if countHits12 > countHits21:
        return -1
    elif countHits12 < countHits21:
        return 1
    else:
        return 0


def ring1_isoutside_ring2_cmp(ring1,ring2,outside_point,bdry_path):

    if ring1 is ring2:
        return 0
    r1nL0 = ring1.nL2bdry_a
    r1nL1 = ring1.nL2bdry_b
    r2nL0 = ring2.nL2bdry_a
    r2nL1 = ring2.nL2bdry_b
    r1_cant_be_outside_r2 = r2_cant_be_outside_r1 = False
    r1_l2b0_inters = pathXpathIntersections(r1nL0,ring2.path)
    r1_l2b1_inters = pathXpathIntersections(r1nL1,ring2.path)
    if r1_l2b0_inters or r1_l2b1_inters:
        r1_cant_be_outside_r2 =  True
    r2_l2b0_inters = pathXpathIntersections(r2nL0,ring1.path)
    r2_l2b1_inters = pathXpathIntersections(r2nL1,ring1.path)
    if r2_l2b0_inters or r2_l2b1_inters:
        r2_cant_be_outside_r1 =  True

    if r1_cant_be_outside_r2 and not r2_cant_be_outside_r1:
        return -1
    elif not r1_cant_be_outside_r2 and r2_cant_be_outside_r1:
        return 1
    elif r1_cant_be_outside_r2 and r2_cant_be_outside_r1:
        p2d = [ring1.path,ring2.path]+[r1nL0,r1nL1,r2nL0,r2nL1]
        p2dc = ['green','red']+['blue','purple']+['yellow','orange']
        dis(p2d,p2dc)
        raise Exception("Cyclic dependency detected.")
    else:
        if len(ring1.path)>1:
            r1_pt = ring1.path[1].start #a point thats not right on the end
        else:
            r1_pt = ring1.path.point(.5)
        if len(ring2.path)>1:
            r2_pt = ring2.path[1].start
        else:
            r2_pt = ring2.path.point(.5)

        r1_must_be_outside_r2 = path_encloses_pt(
            r1_pt, outside_point, ring2.path_around_bdry(bdry_path))
        r2_must_be_outside_r1 = path_encloses_pt(
            r2_pt, outside_point, ring1.path_around_bdry(bdry_path))

        if r1_must_be_outside_r2 and not r2_must_be_outside_r1:
            return 1
        elif not r1_must_be_outside_r2 and r2_must_be_outside_r1:
            return -1
        elif not r1_must_be_outside_r2 and not r2_must_be_outside_r1:
            return 0
        else:
            raise Exception("This case should never be reached.")


class ClosedPair(object):

    def __init__(self, ring_list, outside_point, inner, outer,  contents=[]):
        self.outside_point = outside_point  # a point known to be outside outer
        self.inner_index = inner
        self.outer_index = outer
        self.contents = contents  # indices of open rings contained in pair
        self.contents_psorting = None
        self.ring_list = ring_list
        self.is_core = (inner == 'core')

        self.inner = 'core' if self.is_core else ring_list[inner]
        self.outer = ring_list[outer]

    def intersect(self, path):
        """returns the T values at which path intersects inner or outer.

        This is only used for debugging, the `ClosedPair.contains` may
        not function properly if there are intersections.
        """
        outer_Ts = [T for (T, _, _), _ in path.intersect(self.outer.path)]
        if self.is_core:
            return outer_Ts
        inner_Ts = [T for (T, _, _), _ in path.intersect(self.inner.path)]
        return inner_Ts + outer_Ts

    def contains(self, or_index):
        """note all intersection between rings have already been removed"""
        oring = self.ring_list[or_index]

        # take a test point from somewhere in the middle of the open ring
        pt = oring.path.point(0.5)

        if self.is_core:
            if oring.maxR > self.outer.maxR:
                return False
            return path_encloses_pt(pt, self.outside_point, self.outer.path)

        if oring.maxR > self.outer.maxR or oring.minR < self.inner.minR:
            return False
        return path_encloses_pt(pt, self.outside_point, self.outer.path) and \
               not path_encloses_pt(pt, self.outside_point, self.inner.path)

    def __repr__(self):
        return f"CP({self.inner_index}, {self.outer_index}): {self.contents}"


def debug_unlocated_rings_and_raise_error(unlocated_open_ring_indices,
                                          ring_list, closed_pairs):
    error_message = (
            "\n\nERROR: There are one or more open rings that appear "
            "not to be contained between any pair of closed rings.  "
            "This probably means that said open ring(s) intersect "
            "with closed rings.\n"
            "Note that svg-dendro will try to remove these "
            "intersections if the `rings_may_contain_intersections` "
            "option is set to True.  Currently "
            "`rings_may_contain_intersections` is set to %s."
            "\nWe'll now generate some SVGs to show any closed rings "
            "which intersect with the unlocated open rings.\n"
            "These SVGs can be found at:\n"
            "" % opt.rings_may_contain_intersections
    )

    for problem_ring_index in unlocated_open_ring_indices:
        problem_ring = ring_list[problem_ring_index]
        problem_path = problem_ring.path

        for cp in closed_pairs:
            intersections = cp.intersect(problem_path)
            if not intersections:
                continue

            # create SVG showing intersection points
            debug_svg_path = os.path.join(
                opt.output_directory_debug,
                f'{cp.inner_index}-{cp.outer_index}.svg'
            )
            paths_to_show = [problem_path, cp.outer.path]
            colors = 'br'
            if not cp.is_core:
                paths_to_show.append(cp.inner.path)
                colors += 'g'
            wsvg(paths_to_show,
                 colors=colors,
                 filename=debug_svg_path,
                 nodes=[problem_path.point(T) for T in intersections])
            error_message += f"{debug_svg_path}\n"
    raise Exception(error_message)


def sort_rings(ring_list, om_pickle_file):
    """make list of pairs of consecutive closed rings"""
    basic_output_on.dprint("\nSorting closed rings...",'nr')
    bdry_ring = max(ring_list, key=lambda rg: rg.maxR)
    outside_point = bdry_ring.center + 2*bdry_ring.maxR  # is outside all rings

    sorted_closed_ring_indices = ['core']
    sorted_closed_ring_indices += \
        sorted([rl_ind for rl_ind, r in enumerate(ring_list) if r.isClosed()],
               key=lambda idx: ring_list[idx].maxR)

    closed_pairs = [ClosedPair(ring_list,
                               outside_point,
                               sorted_closed_ring_indices[k-1],
                               sorted_closed_ring_indices[k])
                    for k in range(1, len(sorted_closed_ring_indices))]

    # Find the lines to the boundary and the path given
    if not use_alternative_sorting_method:
        center = ring_list[0].center
        d = 1.5 * bdry_ring.maxR
        pts = [center - d + d*1j, center - d - d*1j,
               center + d - d*1j, center + d + d*1j]
        rectangle_containing_bdry = \
            Path(*[Line(pts[i], pts[(i+1) % 4]) for i in range(4)])
        for r in ring_list:
            if not r.isClosed():
                r.findLines2Bdry(rectangle_containing_bdry)

    # figure out which open (incomplete) rings live between which closed rings
    basic_output_on.dprint(
        "Done, closed rings sorted.\nNow determining which open rings "
        "lie between which closed pairs of rings...", 'nr'
    )
    start_time = current_time()
    unlocated_open_ring_indices = \
        set(i for i, r in enumerate(ring_list) if not r.isClosed())

    for cp in closed_pairs:
        cp.contents = [r_idx for r_idx in unlocated_open_ring_indices
                       if cp.contains(r_idx)]
        unlocated_open_ring_indices -= set(cp.contents)

    # there should not be any unlocated open ring indices
    # in case there are, this is likely caused by intersections
    if unlocated_open_ring_indices:
        debug_unlocated_rings_and_raise_error(
            unlocated_open_ring_indices, ring_list, closed_pairs)

    basic_output_on.dprint(
        "\rFinished locating open rings. Total time elapsed: %s"
        "" % format_time(current_time()-start_time))

#    ###DEBUG ONLY TEST slideshow (of which rings are put in which closed ring pairs)
#    basic_output_on.dprint("creating slideshow of which rings are located between which closed ring pairs...",'nr')
#    from os import path as os_path
#    from options4rings import output_directory
#    from andysSVGpathTools import svgSlideShow
#    save_dir = os_path.join(output_directory,'debug','slideshow_closed_pair_inclusions')
#    pathcolortuplelist = []
#    paths = [ring.path for ring in ring_list]
#    for cp in closed_pairs:
#        colors = ['yellow']*len(paths)
#        if cp.inner_index !='core':
#            colors[cp.inner_index] = 'red'
#        colors[cp.outer_index] = 'green'
#        for i in cp.contents:
#            colors[i] = 'blue'
#        pathcolortuplelist.append((paths,colors))
#    svgSlideShow(pathcolortuplelist,save_directory=save_dir,clear_directory=True,suppressOutput=not basic_output_on.b)
#    ###End of DEBUG ONLY TEST slideshow (of which rings are put in which closed ring pairs)

    # sort the open rings inside each pair of closed rings
    start_time = current_time()
    
    ordering_matrices_pickle_extant = False
    if look_for_user_sort_input:
        try:
            ordering_matrices = pickle.load(open(om_pickle_file, "rb"))
            ordering_matrices_pickle_extant = True
        except:
            from warnings import warn
            warn("No ordering matrices pickle file found.");sleep(1)
    if use_alternative_sorting_method:
        def ring_index_cmp(idx1, idx2):
            return ring1_isoutside_ring2_cmp_alt(ring_list, idx1, idx2)
    else:
        def ring_index_cmp(idx1, idx2): 
            return ring1_isoutside_ring2_cmp(
                ring_list[idx1], ring_list[idx2], outside_point, bdry_ring.path)
    basic_output_on.dprint("Sorting open rings inside each cp...")
    start_time_cp_sorting = current_time()
    et = 0
    cp_oms = []
    flag_count = 0
    num_seg_pairs2check = sum([sum([len(ring_list[i].path)*(len(ring_list[j].path)-1)/2 for (i,j) in combinations(cp.contents,2)]) for cp in closed_pairs])
    num_seg_pairs_checked = 0
    for k,cp in enumerate(closed_pairs):
        if not len(cp.contents):
            if not ordering_matrices_pickle_extant:
                cp_oms.append([])
            continue
        if ordering_matrices_pickle_extant:
            om = ordering_matrices[k]
            #THIS BLOCK IS REPLACED BELOW (DELETE BLOCK)...
#            for i in len(om):
#                for j in len(om):
#                    if isnan(om[i,j]):
#                        om[i,j] = ask_user_to_sort(i,j,ring_list,make_svg=True,ask_later=False)
#                        om[j,i] = -om[i,j] #...THIS BLOCK IS REPLACED BELOW (DELETE BLOCK)
            tmp_time = current_time()
            for i,j in transpose(where(isnan(om))):
                if i<j:
                    om[i,j] = ask_user_to_sort(cp.contents[i], 
                                                cp.contents[j],
                                                ring_list,make_svg=True, 
                                                ask_later=False)
                    om[j,i] = -om[i,j]
            start_time_cp_sorting -= current_time() - tmp_time 
        else:
            om = createOrderingMatrix(cp.contents,ring_index_cmp)
            cp_oms.append(om)
        try:
            assert not any(flattenList(isnan(om)))
        except AssertionError:
            flag_count += 1
            pass
        num_seg_pairs_checked += sum(
            len(ring_list[i].path) * (len(ring_list[j].path) - 1) / 2
            for i, j in combinations(cp.contents, 2)
        )

        try:  # lazy fix for test cases where num_seg_pairs2check==0
            percent_complete = num_seg_pairs_checked/num_seg_pairs2check
        except ZeroDivisionError:
            percent_complete = k/len(closed_pairs)
            pass

        if not flag_count:
            psorting = topo_sorted(cp.contents, ring_index_cmp, ordering_matrix=om)

            cp.contents = [cp.contents[index] for index in flattenList(psorting)]
            cp.contents_psorting = psorting
        et_tmp = current_time() - start_time_cp_sorting
        
        if et_tmp > et + 3:
            et = et_tmp
            etr = (1-percent_complete)*et/percent_complete
            basic_output_on.dprint("%s percent complete. Time Elapsed = %s | ETR = %s"%(int(percent_complete*100),format_time(et),format_time(etr)))

    #Output problem cases for manual sorting
    from options4rings import output_directory
    from os import path as os_path
    from andysmod import output2file
    manual_sort_csvfile = os_path.join(output_directory,"interactive_sorting",ring_list[0].svgname,"manual_comparisons.csv")
    str_out = ''
    if flag_count:
        pickle.dump(cp_oms, open(om_pickle_file, "wb"))
        output2file(str_out,filename=manual_sort_csvfile,mode='w')
        for k,om in enumerate(cp_oms):
            cp = closed_pairs[k]
            problem_pairs = [(cp.contents[i],cp.contents[j]) for i,j in transpose(where(isnan(om))) if i<j]
            problem_pairs = sorted(problem_pairs,key=itemgetter(0))
            for (idx_i,idx_j) in problem_pairs:
                str_out+='%s,%s,\n'%(idx_i,idx_j)
            output2file(str_out,filename=manual_sort_csvfile,mode='a')

        raise Exception("There are %s rings pairs that need to be manually sorted.  Please set 'look_for_user_sort_input=True' and run this svg again.  Note: When you run again, there will be an interactive interface to help you sort, but it may be easier to manually enter the needed comparisons in\n%s"%(flag_count,manual_sort_csvfile))
    basic_output_on.dprint("Done with inner ring sorting (in %s).  Finished with %s error flags."%(format_time(current_time()-start_time),flag_count))

    # Note: sort_lvl info records the number of other rings in the same 
    # sort level, so in the future I can output psort_index values as 3.0, 3.1, etc
    ring_sorting = [cp.contents+[cp.outer_index] for cp in closed_pairs]
    ring_sorting = flattenList(ring_sorting)
    sort_lvl_info = []
#    for cp in closed_pairs:
#        for sort_lvl in cp.contents_psorting:
#            sort_lvl_info += [len(sort_lvl)]*len(sort_lvl)
#        sort_lvl_info += [1]  # for outer ring in cp
    return ring_sorting, sort_lvl_info