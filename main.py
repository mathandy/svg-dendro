# External Dependencies
from __future__ import division
import os
import argparse
import _pickle as pickle
from time import time as current_time
import numpy as np
from svgpathtools import wsvg

# Internal Dependencies
from andysmod import format_time
from misc4rings import displaySVGPaths_transects, plotUnraveledRings
from transects4rings import (generate_inverse_transects,
                             generate_sorted_transects)
from svg2rings import svg2rings
from fixsvg import fix_svg
import options4rings as opt


def svgtree(filepath, error_list):

    file_start_time = current_time()
    svg_name = os.path.splitext(os.path.basename(filepath))[0]

    # Name pickle Files
    pickle_file = os.path.join(
        opt.pickle_dir, svg_name + "-ring_list.p")
    sorted_pickle_file = os.path.join(
        opt.pickle_dir, svg_name + "-sorted-ring_list.p")
    om_pickle_file = os.path.join(
        opt.pickle_dir, svg_name + "-ordering_matrix.p")

    # load rings and compute center (possible from
    loaded_from_pickle = False
    loaded_from_sorted_pickle = False
    if not opt.ignore_extant_sorted_pickle_file:
        # load ring_list and center from sorted pickle file
        try:
            ring_list, center = pickle.load(open(sorted_pickle_file, "rb"))
            loaded_from_sorted_pickle = True
        except:
            pass

    # determine if pickle file exists, if it does,
    if not opt.ignore_extant_pickle_file and not loaded_from_sorted_pickle:
        # load ring_list and center from unsorted pickle file
        try:
            ring_list, center = pickle.load(open(pickle_file, "rb"))
            loaded_from_pickle = True
        except:
            pass

    if not loaded_from_pickle and not loaded_from_sorted_pickle:
        # If pickle file doesn't exist, create one, and
        # store ring_list and center in it
        center, ring_list = svg2rings(filepath)
        opt.basic_output_on.dprint("Pickling ring_list... ", 'nr')
        pickle.dump((ring_list, center), open(pickle_file, "wb"))
        opt.basic_output_on.dprint('pickling complete -> ' + pickle_file)
        opt.basic_output_on.dprint("Done.")

    ####################################################################
    # hack to record svg names in rings ################################
    ####################################################################
    for ring in ring_list:
        ring.svgname = svg_name

    if not opt.assume_svg_is_fixed:
        fix_svg(ring_list, center, svg_name)

    ####################################################################
    # Sort #############################################################
    ####################################################################
    # sort ring_list from innermost to outermost and record sort index
    if opt.sort_rings_on:
        if not loaded_from_sorted_pickle:
            opt.basic_output_on.dprint(
                "Attempting to sort ring_list.  This could take "
                "a minute (or thirty)...", 'nr')

            # find sorting of ring_list
            from sorting4rings import sort_rings
            ring_sorting, sort_lvl_info = sort_rings(ring_list, om_pickle_file)
            opt.basic_output_on.dprint("Done sorting ring_list.")

            # record sort index
            for i, r_index in enumerate(ring_sorting):
                ring_list[r_index].sort_index = i

            # pickle "sorted" ring_list (not really sorted, but sort_index's
            # are recorded)
            opt.basic_output_on.dprint("Pickling sorted ring_list... ", 'nr')
            pickle.dump((ring_list, center), open(sorted_pickle_file, "wb"))
            opt.basic_output_on.dprint('pickling complete -> ' +
                                       sorted_pickle_file)

    ####################################################################
    # Generate and Output Transects ####################################
    ####################################################################

    # generate transects
    skipped_angle_indices = []
    tmp = "Generating the {} requested transects...".format(opt.N_transects)
    opt.basic_output_on.dprint(tmp, 'nr')
    if opt.N_transects > 0:
        if opt.use_ring_sort_4transects:
            if opt.generate_evenly_spaced_transects:

                angles = np.linspace(0, 1, opt.N_transects, endpoint=False)
                bdry_ring = max(ring_list, key=lambda r: r.maxR)
                bdry_length = bdry_ring.path.length()
                Tvals = [bdry_ring.path.ilength(s * bdry_length) for s in angles]

                data, data_indices, skipped_angle_indices = \
                    generate_inverse_transects(ring_list, Tvals)

                opt.basic_output_on.dprint(
                    f"{len(data)} / {opt.N_transects} evenly spaced transects "
                    f"successfully generated (skipped "
                    f"{opt.N_transects - len(data)})."
                )
            else:
                data, data_indices, angles = generate_sorted_transects(
                    ring_list, center, angles2use=opt.angles2use)
        else:
            from transects4rings import generate_unsorted_transects
            data, data_indices, angles = \
                generate_unsorted_transects(ring_list, center)
        opt.basic_output_on.dprint("Done generating transects.")

        # show them (this creates an svg file in the output folder)
        if opt.create_SVG_picture_of_transects:
            svg_trans = os.path.join(opt.output_directory,
                                     svg_name + "_transects.svg")
            displaySVGPaths_transects(ring_list, data, angles,
                                      skipped_angle_indices, fn=svg_trans)
            opt.basic_output_on.dprint("\nSVG showing transects generated "
                                       "saved to:\n{}\n".format(svg_trans))

        # Save results from transects
        from transects4rings import save_transect_data, save_transect_summary
        # Name output csv files
        tmp = 'TransectDataFrom-' + svg_name + '.csv'
        outputFile_transects = os.path.join(opt.output_directory, tmp)
        tmp = 'TransectSummary-' + svg_name + '.csv'
        outputFile_transect_summary = os.path.join(opt.output_directory, tmp)
        completed_angles = [x for idx, x in enumerate(angles)
                            if idx not in skipped_angle_indices]
        skipped_angles = [angles[idx] for idx in skipped_angle_indices]
        save_transect_data(outputFile_transects, ring_list, data,
                           data_indices, completed_angles, skipped_angles)
        save_transect_summary(outputFile_transect_summary, ring_list, data,
                              data_indices, completed_angles)

    ####################################################################
    # Compute Ring Areas ###############################################
    ####################################################################
    if opt.find_areas:
        from area4rings import find_ring_areas
        sorted_ring_list = sorted(ring_list, key=lambda rg: rg.sort_index)

        # this also completes incomplete rings
        find_ring_areas(sorted_ring_list, center, filepath)

    ####################################################################
    # Other (optional) stuff ###########################################
    ####################################################################

    # Create SVG showing ring sorting
    if opt.create_SVG_showing_ring_sort:
        opt.basic_output_on.dprint(
            "Attempting to create SVG showing ring sorting...", 'nr')

        from misc4rings import displaySVGPaths_numbered
        tmp = svg_name[:-4] + "_sort_numbered" + ".svg"
        wsvg(paths=[r.path for r in ring_list],
             colors=[r.color for r in ring_list],
             filename=os.path.join(opt.output_directory_debug, tmp),
             text=[str(i) for i in range(len(ring_list))],
             text_path=[r.path for r in ring_list])
        # displaySVGPaths_numbered([r.path for r in ring_list],
        #                          os.path.join(opt.output_directory_debug, tmp),
        #                          [r.color for r in ring_list])
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

    ####################################################################
    # Report Success/Failure of file ###################################
    ####################################################################
    tmp = (filepath, format_time(current_time() - file_start_time))
    opt.basic_output_on.dprint("Success! Completed {} in {}.".format(*tmp))
    opt.basic_output_on.dprint(":)"*50)
    opt.basic_output_on.dprint("<>"*50)
    opt.basic_output_on.dprint("<>"*50)
    opt.basic_output_on.dprint("\n\n")
    error_list.append((filepath, "Completed Successfully."))
    return


def get_user_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_path',
        dest='input_path',
        default=opt.input_path,
        help='The directory or single svg file to input (defaults to {}).  '
             'Use quotes for file/directory locations containing spaces)'
             ''.format(opt.input_path),
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
             'the input svg is a fixed svg output by svg-dendro.',
        # metavar='ASSUME_SVG_IS_FIXED'
        )

    parser.add_argument(
        '-p', '--use_pickle_files',
        # dest='use_pickle_files',
        default= not opt.ignore_extant_sorted_pickle_file,
        action='store_true',
        help='If this flag is included, SVGTree will attempt to save time by '
             'loading the sorting information extracted from an SVG on a '
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
             "test_examples folder:\n{}\nNote: overrides --input_path flag."
             "".format(os.path.join(
            os.getcwd(), 'input', 'examples', 'test_examples')),
    )

    parser.add_argument(
        '--reals',
        # dest='stop_on_error',
        default=False,
        action='store_true',
        help="If this flag is included, SVGTree run the real data samples "
             "stored in the real_examples folder:\n{}\n"
             "Note: overrides --input_path and --fakes flags."
             "".format(os.path.join(
            os.getcwd(), 'input', 'examples', 'real_examples')),
    )

    return parser.parse_args()


if __name__ == '__main__':
    user_args = get_user_args()

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

    ####################################################################
    # Check input path #################################################
    ####################################################################
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    if user_args.reals:
        opt.input_path = os.path.join(_repo_root, 'data', 'real_examples')
    elif user_args.fakes:
        opt.input_path = os.path.join(_repo_root, 'data', 'test_examples')
    else:
        opt.input_path = user_args.input_path
        assert (os.path.isfile(opt.input_path) or
                os.path.isdir(opt.input_path)), (
                "You're input directory/file must be a valid "
                "directory (or SVG file).  You could also simply "
                "put your SVG files in the 'input' folder (inside "
                "the SVGTree folder that this code is stored in)."
            )

    ####################################################################
    # Check if output (sub)directories exist, create subdirs if not ####
    ####################################################################
    # Get output directory
    opt.output_directory = user_args.output_directory
    opt.pickle_dir = os.path.join(opt.output_directory, "pickle_files")
    opt.output_directory_debug = os.path.join(opt.output_directory, "debug")
    opt.unsorted_transect_debug_output_folder = os.path.join(
        opt.output_directory, "debug", "transect_slides")
    # assert os.path.exists(opt.output_directory), (
    #         "\n\nThe output_directory given in options does not exist.  "
    #         "To fix this change output_directory in options, or "
    #         "create the folder:\n%s" % opt.output_directory
    # )
    if not os.path.exists(opt.pickle_dir):
        os.makedirs(opt.pickle_dir)
    if not os.path.exists(opt.output_directory_debug):
        os.makedirs(opt.output_directory_debug)
    if not os.path.exists(opt.unsorted_transect_debug_output_folder):
        os.makedirs(opt.unsorted_transect_debug_output_folder)

    ####################################################################
    # Batch run all SVG filed in input directory #######################
    ####################################################################
    error_list = []
    if os.path.isdir(opt.input_path):
        svgfiles = [os.path.join(opt.input_path, fn)
                    for fn in os.listdir(opt.input_path) if fn.endswith('.svg')]
    else:
        svgfiles = [opt.input_path]

    for svg in svgfiles:

        #######################################################################
        # Load SVG, extract rings, pickle (or just load pickle if it exists) ##
        #######################################################################
        if svg[len(svg) - 3: len(svg)] == 'svg':
            try:
                print('-' * 40 + '\n' + '~' * 20 +
                      'attempting {}'.format(svg) +
                      '\n' + '-' * 40)

                # Analyze svgfile
                svgtree(svg, error_list)

            except:
                print("-"*75)
                print("!" * 25 + svg + " did not finish successfully.")
                print("Reason:")

                # save error to error_list
                if opt.if_file_throws_error_skip_and_move_to_next_file:
                    from traceback import format_exc
                    from sys import stdout
                    stdout.write(format_exc())
                    error_list.append((svg, format_exc()))
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

    error_log = os.path.join(opt.output_directory, "error_list.txt")
    print("error_list ouput to:\n{}".format(error_log))
    with open(error_log, 'wt') as outf:
        for fn, err in error_list:
            outf.write('#'*50 + '\n')
            outf.write('### ' + fn + '\n')
            outf.write('#'*50 + '\n')
            outf.write(err + '\n'*3)
    print("All done.")
