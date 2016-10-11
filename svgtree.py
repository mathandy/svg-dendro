# External Dependencies
from __future__ import division
from os import listdir, makedirs as os_makedirs, path as os_path, getcwd
from ntpath import basename as nt_path_basename
import argparse
import cPickle as pickle
from time import time as current_time
import numpy as np

# Internal Dependencies
from andysmod import format_time
from misc4rings import displaySVGPaths_transects, plotUnraveledRings
from transects4rings import (generate_inverse_transects,
                             generate_sorted_transects)
from svg2rings import svg2rings
from fixsvg import fix_svg
import options4rings as opt


def get_user_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_directory',
        dest='input_directory',
        default=opt.input_directory,
        help='The directory or single svg file to input (defaults to {}).  '
             'Use quotes for file/directory locations containing spaces)'
             ''.format(opt.input_directory),
        # metavar='INPUT_DIRECTORY'
        )

    parser.add_argument(
        '-o', '--output_directory',
        dest='output_directory',
        default=opt.output_directory,
        help='The directory to store ouput in (defaults to {}).  '
             'Use quotes for directory locations containing spaces)'
             ''.format(opt.output_directory),
        # metavar='OUTPUT_DIRECTORY'
        )

    parser.add_argument(
        '-n', '--n_transects',
        type=int,
        # dest='n_transects',
        default=opt.N_transects,
        help='The number of transects to find (defaults to {}).'
             ''.format(opt.N_transects),
        # metavar='N_TRANSECTS'
        )

    parser.add_argument(
        '-a', '--no_areas',
        # dest='no_areas',
        default=not opt.find_areas,
        action='store_true',
        help='If this flag is included, areas will not be computed.',
        # metavar='NO_AREAS'
        )

    parser.add_argument(
        '-r', '--random_transects',
        # dest='random_transects',
        default=not opt.generate_evenly_spaced_transects,
        action='store_true',
        help='If this flag is included, the transects found will be '
             'random as oppposed to evenly spaced.'
             'NOTE:  BROKEN... this is currently broken, but is a '
             'simple feature.  If anyone wants to use it, let me know.',
        # metavar='RANDOM_TRANSECTS'
        )

    parser.add_argument(
        '-f', '--assume_fixed',
        # dest='assume_svg_is_fixed',
        default=opt.assume_svg_is_fixed,
        action='store_true',
        help='If this flag is included, the there will be no pre-processing '
             'done on the svg paths.  This should generally only be used if '
             'the input svg is a fixed svg output by SVGTree.',
        # metavar='ASSUME_SVG_IS_FIXED'
        )

    parser.add_argument(
        '-p', '--use_pickle_files',
        # dest='use_pickle_files',
        default= not opt.ignore_extant_sorted_pickle_file,
        action='store_true',
        help='If this flag is included, SVGTree will attempt to save time by '
             'loading the sorting informationing extracted from an SVG on a '
             'previous run.',
    )

    parser.add_argument(
        '-l', '--look_for_user_sort_input',
        # dest='look_for_user_sort_input',
        default=opt.look_for_user_sort_input,
        action='store_true',
        help='If this flag is included, SVGTree will look for user responses '
             'added to the interactive sorting ouput from a previous run.',
        # metavar='LOOK_FOR_USER_SORT_INPUT'
    )

    parser.add_argument(
        '-e', '--stop_on_error',
        # dest='stop_on_error',
        default= not opt.if_file_throws_error_skip_and_move_to_next_file,
        action='store_true',
        help="If this flag is included, SVGTree will stop and raise errors "
             "when they are encountered as opposed to moving on to the next "
             "SVG file.",
        # metavar='STOP_ON_ERROR'
    )

    parser.add_argument(
        '--fakes',
        # dest='stop_on_error',
        default=False,
        action='store_true',
        help="If this flag is included, SVGTree run fake data samples in the "
             "test_examples folder:\n{}\nNote: overrides --input_directory flag."
             "".format(os_path.join(getcwd(), 'input', 'examples', 'test_examples')),
    )

    parser.add_argument(
        '--reals',
        # dest='stop_on_error',
        default=False,
        action='store_true',
        help="If this flag is included, SVGTree run the real data samples "
             "stored in the real_examples folder:\n{}\n"
             "Note: overrides --input_directory and --fakes flags."
             "".format(os_path.join(getcwd(), 'input', 'examples', 'real_examples')),
    )

    return parser.parse_args()


if __name__ == '__main__':
    user_args = get_user_args()

    # Get input directory
    if user_args.reals:
        opt.input_directory = os_path.join(getcwd(), 'input', 'examples', 'real_examples')
    elif user_args.fakes:
        opt.input_directory = os_path.join(getcwd(), 'input', 'examples', 'test_examples')
    else:
        try:
            assert (os_path.isfile(user_args.input_directory) or
                    os_path.isdir(user_args.input_directory))
        except AssertionError:
            raise IOError("You're input directory/file must be a valid "
                          "directory (or SVG file).  You could also simply "
                          "put your SVG files in the 'input' folder (inside "
                          "the SVGTree folder that this code is stored in).")

        opt.input_directory = user_args.input_directory

    # Get output directory
    assert os_path.isdir(user_args.output_directory)
    opt.output_directory = user_args.output_directory

    # Get optional parameters
    opt.N_transects = int(user_args.n_transects)
    opt.find_areas = user_args.no_areas
    opt.generate_evenly_spaced_transects = not user_args.random_transects
    # opt.use_ring_sort_4transects = not user_args.random_transects
    opt.assume_svg_is_fixed = user_args.assume_fixed
    opt.ignore_extant_pickle_file = not user_args.use_pickle_files
    opt.ignore_extant_sorted_pickle_file = not user_args.use_pickle_files
    opt.look_for_user_sort_input = user_args.look_for_user_sort_input
    opt.if_file_throws_error_skip_and_move_to_next_file = not user_args.stop_on_error


def svgtree(svgfile, error_list):

    file_start_time = current_time()

    SVGfileLocation = os_path.join(opt.input_directory, svgfile)
    svgname = nt_path_basename(SVGfileLocation)[0:-4]

    # Name pickle Files
    pickle_file = os_path.join(opt.pickle_dir, svgname + "-ring_list.p")
    sorted_pickle_file = os_path.join(opt.pickle_dir, svgname + "-sorted-ring_list.p")
    om_pickle_file = os_path.join(opt.pickle_dir, svgname + "-ordering_matrix.p")
    # tmp = 'DataFrom-'+ svgname +'_failed_rings.csv'
    # outputFile_failed_rings = os_path.join(opt.output_directory, tmp)
    # tmp = 'DataFrom-' + svgname + '.csv'
    # outputFile = os_path.join(opt.output_directory, tmp)

    # determine if pickle file exists, if it does,
    # load ring_list and center from it
    if opt.ignore_extant_pickle_file:
        pickle_file_exists = False
    else:
        pickle_file_exists = True
        try:
            ring_list, center = pickle.load(open(pickle_file, "rb"))
        except:
            pickle_file_exists = False

    # determine if sorted pickle file exists, if it does,
    # load ring_list and center from it (instead of unsorted pickle)
    if opt.ignore_extant_sorted_pickle_file:
        sorted_pickle_file_exists = False
    else:
        sorted_pickle_file_exists = True
        try:
            ring_list, center = pickle.load(open(sorted_pickle_file, "rb"))
        except:
            sorted_pickle_file_exists = False

    # If pickle file doesn't exist, create one, and
    # store ring_list and center in it
    if not (pickle_file_exists or sorted_pickle_file_exists):
        center, ring_list = svg2rings(SVGfileLocation)
        opt.basic_output_on.dprint("Pickling ring_list... ", 'nr')
        pickle.dump((ring_list, center), open(pickle_file, "wb"))
        opt.basic_output_on.dprint('pickling complete -> ' + pickle_file)
        opt.basic_output_on.dprint("Done.")

###############################################################################
###fix to record svg names in rings ###########################################
###############################################################################
    for ring in ring_list:
        ring.svgname = svgfile[:-4]

    if not opt.assume_svg_is_fixed:
        fix_svg(ring_list, center, svgfile)

###############################################################################
###Sort #######################################################################
###############################################################################
    # sort ring_list from innermost to outermost and record sort index
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

    # generate transects
    skipped_angle_indices = []
    tmp = "Generating the {} requested transects...".format(opt.N_transects)
    opt.basic_output_on.dprint(tmp, 'nr')
    if opt.N_transects > 0:
        if opt.use_ring_sort_4transects:
            if opt.generate_evenly_spaced_transects:
                angles = np.linspace(0, 1, opt.N_transects+1)[:-1]
                bdry_ring = max(ring_list, key=lambda r: r.maxR)
                bdry_length = bdry_ring.path.length()
                Tvals = [bdry_ring.path.ilength(s * bdry_length) for s in angles]

                tmp = generate_inverse_transects(ring_list, Tvals)
                data, data_indices, skipped_angle_indices = tmp

                num_suc = len(data)
                nums = (num_suc, opt.N_transects, opt.N_transects - num_suc)
                trmes = ("%s / %s evenly spaced transects successfully "
                         "generated (skipped %s)." % nums)
                opt.basic_output_on.dprint(trmes)
            else:
                tmp = generate_sorted_transects(ring_list, center,
                                                angles2use=opt.angles2use)
                data, data_indices, angles = tmp
        else:
            from transects4rings import generate_unsorted_transects
            tmp = generate_unsorted_transects(ring_list, center)
            data, data_indices, angles = tmp
        opt.basic_output_on.dprint("Done generating transects.")

        # show them (this creates an svg file in the output folder)
        if opt.create_SVG_picture_of_transects:
            svgname = svgfile[0:len(svgfile)-4]
            svg_trans = os_path.join(opt.output_directory,
                                     svgname + "_transects.svg")
            displaySVGPaths_transects(ring_list, data, angles,
                                      skipped_angle_indices, fn=svg_trans)
            opt.basic_output_on.dprint("\nSVG showing transects generated "
                                       "saved to:\n{}\n".format(svg_trans))


        # Save results from transects
        from transects4rings import save_transect_data, save_transect_summary
        # Name output csv files
        tmp = 'TransectDataFrom-' + svgname + '.csv'
        outputFile_transects = os_path.join(opt.output_directory, tmp)
        tmp = 'TransectSummary-' + svgname + '.csv'
        outputFile_transect_summary = os_path.join(opt.output_directory, tmp)
        completed_angles = [x for idx, x in enumerate(angles)
                            if idx not in skipped_angle_indices]
        skipped_angles = [angles[idx] for idx in skipped_angle_indices]
        save_transect_data(outputFile_transects, ring_list, data,
                           data_indices, completed_angles, skipped_angles)
        save_transect_summary(outputFile_transect_summary, ring_list, data,
                              data_indices, completed_angles)

###############################################################################
###Compute Ring Areas #########################################################
###############################################################################
    if opt.find_areas:
        from area4rings import find_ring_areas
        sorted_ring_list = sorted(ring_list, key=lambda rg:rg.sort_index)

        # this also completes incomplete rings
        find_ring_areas(sorted_ring_list, center, svgfile)

###############################################################################
###Other (optional) stuff #####################################################
###############################################################################

    # Create SVG showing ring sorting
    if opt.create_SVG_showing_ring_sort:
        opt.basic_output_on.dprint("Attempting to create SVG showing ring "
                                   "sorting...", 'nr')

        from misc4rings import displaySVGPaths_numbered
        tmp = svgfile[0:len(svgfile)-4] + "_sort_numbered" + ".svg"
        svgname = os_path.join(opt.output_directory_debug, tmp)
        displaySVGPaths_numbered([r.path for r in ring_list], svgname,
                                 [r.color for r in ring_list])
        opt.basic_output_on.dprint("Done.")



    # test complete ring sort after first sort round
    if opt.visual_test_of_all_ring_sort_on:
        from svg2rings import visual_test_of_ring_sort
        visual_test_of_ring_sort(ring_list)

    # plot all rings on a plot with x = theta and y = r
    if opt.showUnraveledRingPlot:
        opt.basic_output_on.dprint("Creating unraveled ring plot... ", 'nr')
        plotUnraveledRings(ring_list, center)
        opt.basic_output_on.dprint("Done.  (It should have opened "
                                   "automatically and now be visible.)")

###############################################################################
###Report Success/Failure of file #############################################
###############################################################################
    tmp = (svgfile, format_time(current_time() - file_start_time))
    opt.basic_output_on.dprint("Success! Completed {} in {}.".format(*tmp))
    opt.basic_output_on.dprint(":)"*50)
    opt.basic_output_on.dprint("<>"*50)
    opt.basic_output_on.dprint("<>"*50)
    opt.basic_output_on.dprint("\n\n")
    error_list.append((svgfile, "Completed Successfully."))
    return


###############################################################################
# Check if output (sub)directories exist, create subdirectories if they don't
# exist
mes = ("\n\nThe output_directory given in options does not exist.  To fix this "
      "change output_directory in options, or create the folder:\n"
       "%s" % opt.output_directory)
assert os_path.exists(opt.output_directory), mes
if not os_path.exists(opt.pickle_dir):  # debug folder
    os_makedirs(opt.pickle_dir)
if not os_path.exists(opt.output_directory_debug):  # pickle folder
    os_makedirs(opt.output_directory_debug)


###############################################################################
###Batch run all SVG filed in input directory #################################
###############################################################################
error_list = []
if os_path.isdir(opt.input_directory):
    svgfiles = listdir(opt.input_directory)
else:
    svgfiles = [opt.input_directory]
    opt.input_directory = os_path.pardir(opt.input_directory)

for svgfile in svgfiles:

    # Get name sans extension
    svgname = svgfile[:-4]

###############################################################################
###Load SVG, extract rings, pickle (or just load pickle if it exists) #########
###############################################################################
    if svgfile[len(svgfile) - 3: len(svgfile)] == 'svg':
        try:
            print('-' * 40 + '\n' + '~' * 20 +
                  'attempting {}'.format(svgfile) +
                  '\n' + '-' * 40)

            # Analyze svgfile
            svgtree(svgfile, error_list)

        except:
            print("-"*75)
            print("!" * 25 + svgfile + " did not finish successfully.")
            print("Reason:")

            #save error to error_list
            if opt.if_file_throws_error_skip_and_move_to_next_file:
                from traceback import format_exc
                from sys import stdout
                stdout.write(format_exc())
                error_list.append((svgfile, format_exc()))
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

error_log = os_path.join(opt.output_directory, "error_list.txt")
print("error_list ouput to:\n{}".format(error_log))
with open(error_log, 'wt') as outf:
    for svgname, err in error_list:
        outf.write('#'*50 + '\n')
        outf.write('### ' + svgname + '\n')
        outf.write('#'*50 + '\n')
        outf.write(err + '\n'*3)
print("All done.")