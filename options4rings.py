from andysmod import ConditionalPrint
from os import path as os_path
from os import getcwd
from tempfile import gettempdir

#### Input and Output directories

# Mac/Linux example:
# input_directory = '/Users/Andy/Desktop/input_svgs/'

# Windows example:
# input_directory = "C:\\Users\\Andy\\Desktop\\input_svgs\\"

# Run just two simple examples
input_path = os_path.join(getcwd(), 'input')

# Run all real sample SVGs
# input_directory = os_path.join(getcwd(), 'input', 'examples', 'real_examples', 'problemsome')

### Output directory
output_directory = os_path.join(gettempdir(), 'svg-dendro-output')


###############################################################################
###Easy Run Options ###########################################################
###############################################################################
if_file_throws_error_skip_and_move_to_next_file = True

# 0 is first file, 1 is second, etc.
start_at_file_number = 0

# number of transects to generate; set to zero if none are desired
N_transects = 50

# This finds transects that end at evenly spaced intervals, this is not like
# how they'd be down by hand.  Note that given a particular point on the
# boundary ring, it's possible that no transect exists ending at that point.
# It's also possible that multiple such transects exist -- in the case
# where multiple possible transect segments exist, svgtree will output the
# shortest transect.
generate_evenly_spaced_transects = True

# Find the area
find_areas = True

# This isn't really an option (unless you're sure rings are sorted in SVG)
sort_rings_on = True


###############################################################################
###saved progess options ######################################################
###############################################################################
# This means the code will do as little as possible to check for human error
# which speeds things up, but the input SVG is actually a fixed SVG (output
# by SVGTree), then this is probably a bad idea
assume_svg_is_fixed = False

# If True then creates an SVG of the correct colors etc to save time in the 
# future in case pickle files need to be redone. The fixed svg will be stored 
# in the output folder set above.
create_fixed_svg = True

# folder to store pickle files (which saves some progress in case the program
# needs to be re-run). By default, these pickle files are NOT used.
pickle_dir = os_path.join(output_directory, "pickle_files")
ignore_extant_pickle_file = True
ignore_extant_sorted_pickle_file = True

# slows down creation of pickle file, speeds up post-pickle performance
save_path_length_in_pickle = True  


###############################################################################
###Smoothing options ##########################################################
###############################################################################
# Note: This will smooth kinks in your rings without changing the data 
# significantly, this is important for efficiency (and several other reasons).
smooth_rings = True  # If True, rings will be check for kinks and smoothed
maxjointsize = 3  # smoothing parameter  (must be positive)
tightness = 1.99  # smoothing parameter (must be in (0, 2))

# If smooth_rings = False, this may allow (if set True) inverse transects 
# to be generated... will work assuming all kinks are joints of line segments
rings_may_contain_unremoved_kinks = True  
ignore_unremovable_kinks = True


##############################################################################
###Tolerances (unit=pixels)###################################################
##############################################################################
# The program will tell you if you should change this... this should probably 
# always be same as tol_isApproxClosedPath
tol_isNear = .1  

#  Any incomplete/merging rings with a gap smaller 
# than this will be forcefully/stupidly closed, making this large has the 
# potential to cause problems. Note: this should probably always be same as 
# tol_isApproxClosedPath.
tol_isApproxClosedPath = 0.1  

# This will sometimes control how close two rings need to be to be
# considered intersecting
tol_intersections = 1e-4

# Remove segments from rings shorter than this (and fix path to be
# continuous after removal).
# Note: it's probably a bad idea for `min_relative_segment_length` to
# not be zero
# Note: smaller segments may be created in the smoothing process (and
# will not removed)
min_relative_segment_length = 0
min_absolute_segment_length = 1e-4


###############################################################################
###Other Options ##############################################################
###############################################################################
# This should be True unless you are sure that no paths in the svg have 
# intersections.  Note: This takes a considerable amount of time to run (though 
# the results will be stored in a pickle file for use in any subsequent runs).
rings_may_contain_intersections = False  

# If True then if the orientation of any paths cannot be determined automatically, 
# will wait for input from user (after outputing an svg to help user decide).
manually_fix_orientations = True

# If paths are not close to being convex (and orientation fails to be determined
# automatically and manually_fix_orientations = False) then this assumption will
# be used.  If set to False, then assumes paths were traced in Clockwise fashion.
when_orientation_cannot_be_determined_assume_CCW = False

# Sometimes the user will be asked for input based on an output (regarding 
# sorting or orientation issues).  If true, the program will attemp to display 
# the output svg in the user's default web browser.  This can be convenient, but 
# sometimes scroll/zoom issues arrise in the browser and the user will want to 
# just open the file in Illustrator or Inkscape anyways.
try_to_open_svgs_in_browser = True

# Trims paths with high curvature ends
remove_curly_ends = True

# Ignore curly ends where entire segment violates curvature tolerance
# ignore_long_curls = True

# If necessary, stop and ask user if curly end can be cropped
# manually_curly_end = True

# (user assistance required unless force_remove_self_intersections) if true, 
# the program check for self-intersections in open rings and if found will 
# stop to ask the user what to do to fix them.
remove_self_intersections = True  

# If true will assume user always answers yes.
force_remove_self_intersections = True 

# Check if rings is improperly closed so that last segment is outward or
# inward of the first segment
# NOTE: this may be buggy and should maybe be False
check4overlappingends = False

# To remove rings too small to be intentional
remove_inappropriately_short_rings = True
appropriate_ring_length_minimum = 2.5 * tol_isNear

dont_remove_closed_inappropriately_short_rings = True
create_svg_highlighting_inappropriately_short_rings = True


###############################################################################
###Debug options: (Note: These will slow down the program significantly.  They 
# should all be False unless you are debugging code.)
###############################################################################
# folder to store debugging output in
output_directory_debug  = os_path.join(output_directory, "debug")
debugging_mode_on = False

# This produces and SVG slideshow showing the ring ordering found.
visual_test_of_all_ring_sort_on = False 
create_SVG_showing_ring_sort = False
sort_debug_mode_on = False
sort_debug_3_on = False
transect_debug_mode_on = False
transect_debug_mode_output_folder = os_path.join(output_directory, "debug")

# This produces and SVG slideshow showing how the transect is formed.  
# You should set N_transects=1 above when using this.
unsorted_transect_debug_on = False 
unsorted_transect_debug_output_folder = os_path.join(output_directory, "debug", "transect_slides")


###############################################################################
###Transect Options ###########################################################
###############################################################################

use_ring_sort_4transects = True

# Sometimes the inverse transects are not unique as the rings are drawn
warn_if_not_unique = False 

# Set this to use specific angles,noting angles should be in [0,1), 
# e.g. angles2use = [0.0435961242631, 0.981667615288, 0.773696429364]
angles2use = None  

# Some inverse transects are impossible -- see example 
# "when inverse transect is impossible.svg" -- unless they cross a ring in a 
# non-normal way
accept_transect_crossings = False 

# only applies to evenly spaced transects
skip_transects_that_dont_exist = True 

# if_transect_fails_continue = False # (NOT IMPLEMENTED) Only applie
create_SVG_picture_of_transects = True


###############################################################################
###Output SVG options #########################################################
###############################################################################
stroke_width_default = 0.1
outputTroubledCPs = True


###############################################################################
###Area options ###############################################################
###############################################################################
# Create SVG showing "completed" paths used for area computation
# These are stored in the debug folder (inside the output folder)
create_SVG_showing_area_paths = True 


###############################################################################
###console output options #####################################################
###############################################################################
showCurrentFilesProgress = ConditionalPrint(True)
showUnraveledRingPlot = False
full_output_on = ConditionalPrint(False)
warnings_output_on = ConditionalPrint(True)
closednessCheck_output_on = ConditionalPrint(True)
basic_output_on = ConditionalPrint(True)
colorcheck_output_on = ConditionalPrint(False)
intersection_removal_progress_output_on = ConditionalPrint(True)
show_transect_progress = ConditionalPrint(True) # Only applies to inverse transects.


###############################################################################
###Sorting options ############################################################
###############################################################################
# Sometimes the automatic sorting has issues due to overlapping paths in input 
# svg file.  You can check for issues using the visual_test_of_all_ring_sort_on 
# option, or you can just fix the sorting manually with this feature.
manually_fix_sorting = True
look_for_user_sort_input = False


###############################################################################
###Alternative Sorting Method #################################################
###############################################################################
use_alternative_sorting_method = True

# This controls how many test normal lines are used to determine whether a ring 
# is above/below another ring (only used for sets of rings that were failed to 
# be sorted by transect data)
alt_sort_N = 10 

# The percentage of transects that must disagree with the majority (regarding 
# sorting) in order for the user to be asked (set to 0 to always ask the user)
percentage_for_disagreement = 0.15 


###############################################################################
###Colors #####################################################################
###############################################################################
colordict = {'center':'#0000FF'}  # blue
colordict.update({'complete':'#FF0000'})  # red
colordict.update({'incomplete':'#FFFF00'})  # yellow
colordict.update({'boundary':'#00FF00'})  # green
colordict.update({'safe1':'#FF9900'})  # used by program to highlight problems.
colordict.update({'safe2':'#00FFCC'})  # used by program to highlight problems.
auto_fix_ring_colors = True
