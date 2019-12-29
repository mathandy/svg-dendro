#External Module Dependencies
from xml.dom import minidom
import re
from time import time as current_time
#Internal Module Dependencies
from misc4rings import   isCCW, closestColor, isApproxClosedPath
from andysmod import format_time, Radius
from andysSVGpathTools import polylineStr2pathStr, isClosedPathStr
from rings4rings import Ring
from svgpathtools import parse_path, Path, disvg
#Options
from options4rings import colordict
import options4rings as opt


def askUserOrientation():
    try: input = raw_input
    except NameError: pass
    dec = input("Enter 'y' or 'n' to specify orientation, or \n"
                "enter 'r' to ignore this path and not include it in the "
                "fixed svg that will be output, or\n"
                "enter 'e' to terminate this session: ")
    if dec=='y':
        return True
    elif dec== 'n':
        return False
    elif dec== 'r':
        return 'remove'
    elif dec== 'e':
        raise Exception("User-forced exit.")
    else:
        askUserOrientation()


def svg2rings(SVGfileLocation):
    global already_warned_having_trouble_extracting_ring_colors
    already_warned_having_trouble_extracting_ring_colors = False

    def getStroke(elem): #get 'stroke' attribute fom xml object
        troubleFlag=False
        stroke = elem.getAttribute('stroke') #sometimes this works
        if stroke=='':
            style = elem.getAttribute('style')
            hexstart = style.find('stroke')
            if hexstart==-1:
                troubleFlag=True
            else:
                temp = style[hexstart:]
                try:
                    stroke = re.search(re.compile('\#[a-fA-F0-9]*'),temp).group()
                except:
                    troubleFlag=True
                    stroke=''

        if troubleFlag:
            global already_warned_having_trouble_extracting_ring_colors
            if not already_warned_having_trouble_extracting_ring_colors:
                already_warned_having_trouble_extracting_ring_colors = True
                opt.warnings_output_on.dprint("Warning: Having trouble extracting hex colors from svg.  Hopefully this will not matter as the palette check will fix the colors.")
        return stroke.upper()




    example_center = r'<line fill="none" stroke="#0000FF" stroke-width="0.15" x1="246.143" y1="380.017" x2="246.765" y2="380.856"/>'

    doc = minidom.parse(SVGfileLocation)  # parseString also exists
    #Find the center
    counter = 0
    centerFound = False
    for elem in doc.getElementsByTagName('line'):
        if getStroke(elem) == colordict['center']:
            center = 0.5*float(elem.getAttribute('x1'))+0.5*float(elem.getAttribute('x2')) + 0.5*float(elem.getAttribute('y1'))*1j +0.5*float(elem.getAttribute('y2'))*1j
            rad = Radius(center)
            centerFound = True
            break
        else:
            counter += 1
    if counter>0 and not centerFound:
        opt.warnings_output_on.dprint("[Warning:] No line objects in the svg were found matching the center color (%s).  Now searching for lines of a color closer to center color than other colors."%counter)
        for elem in doc.getElementsByTagName('line'):
            elem_stroke = getStroke(elem)
            if len(elem_stroke) == 0:
                opt.warnings_output_on.dprint(
                    '[Warning:] stroke has no length -- make a "stroke" '
                    'attribute is included and no CSS classes are being used.')
            elif closestColor(getStroke(elem),colordict) == colordict['center']:
                center = 0.5*float(elem.getAttribute('x1'))+0.5*float(elem.getAttribute('x2')) + 0.5*float(elem.getAttribute('y1'))*1j +0.5*float(elem.getAttribute('y2'))*1j
                rad = Radius(center)
                centerFound = True
                counter -=1
                break
    if counter>0: #center found but counter>0
        opt.warnings_output_on.dprint("[Warning:] There are %s disconnected lines in this SVG not matching the center color.  They will be ignored."%counter)
    try:
        center.real #test if center exists (should be a complex number object)
    except:
        try:
            if counter == 0:

                #Is there a path with the center color?
                for elem in doc.getElementsByTagName('path')+doc.getElementsByTagName('polyline')+doc.getElementsByTagName('polygon'):
                    if getStroke(elem) == colordict['center']:
                        if elem in doc.getElementsByTagName('path'):
                            obtype = 'path'; pathstr = elem.getAttribute('d')
                        elif elem in doc.getElementsByTagName('polyline'):
                           obtype = 'polyline'; pathstr = polylineStr2pathStr(elem.getAttribute('points'))
                        else:
                            obtype = 'polygon'; pathstr = polylineStr2pathStr(elem.getAttribute('points')) + 'z'
                        centerpath = parse_path(pathstr)
                        start,end = centerpath.point(0.25),centerpath.point(0.75)
                        x1,x2,y1,y2 = start.real,end.real,start.imag,end.imag
                        newelem = r'<line fill="none" stroke="%s" stroke-width="0.05" stroke-miterlimit="10" x1="%s" y1="%s" x2="%s" y2="%s"/>'%(colordict['center'],x1,y1,x2,y2)
                        raise Exception("Center of sample should be marked by line of color %s, but no lines are present in svg.  There is a %s with the center color, however.  Open the svg file in a text editor and you should be able to find '%s' somewhere... replace it with '%s'"%(colordict['center'],obtype,elem,newelem))
                else:
                    for elem in doc.getElementsByTagName('path')+doc.getElementsByTagName('polyline')+doc.getElementsByTagName('polygon'):
                        if closestColor(getStroke(elem),colordict) == colordict['center']:
                            if elem in doc.getElementsByTagName('path'):
                                obtype = 'path'; pathstr = elem.getAttribute('d')
                            elif elem in doc.getElementsByTagName('polyline'):
                                obtype = 'polyline'; pathstr = polylineStr2pathStr(elem.getAttribute('points'))
                            else:
                                obtype = 'polygon'; pathstr = polylineStr2pathStr(elem.getAttribute('points')) + 'z'
                            centerpath = parse_path(pathstr)
                            start,end = centerpath.point(0.25),centerpath.point(0.75)
                            x1,x2,y1,y2 = start.real,end.real,start.imag,end.imag
                            newelem = r'<line fill="none" stroke="%s" stroke-width="0.05" stroke-miterlimit="10" x1="%s" y1="%s" x2="%s" y2="%s"/>'%(colordict['center'],x1,y1,x2,y2)
                            raise Exception("Center of sample should be marked by line of color %s, but no lines are present in svg.  There is a path with color close to the center color, however.  Open the svg file in a text editor and you should be able to find '%s' somewhere... replace it with '%s'"%(colordict['center'],obtype,elem,newelem))

                    else:
                        raise Exception('Center of sample should be marked by line of color %s, but no lines are present in svg.  There were no paths or polylines or polygons of a similar color either.  Looks like you did not mark the center. Open your svg in a text editor and search for something that looks like (with different x1, x2, y1, y2 values) \n%s\n'%(colordict['center'],example_center))
        except:

            raise Exception('No center found searching line element with (color) stroke = %s. Open your svg in a text editor and search for something that looks like (with different x1, x2, y1, y2 values) \n%s\n'%(colordict['center'],example_center))

    #Use minidom to extract path strings from input SVG
    opt.basic_output_on.dprint("Extracting path_strings from SVG... ",'nr')
    path_strings = [(p.getAttribute('d'),getStroke(p),p.parentNode.getAttribute('id'),p.toxml()) for p in doc.getElementsByTagName('path')]
    #Use minidom to extract polyline strings from input SVG, convert to path strings, add to list
    path_strings += [(polylineStr2pathStr(p.getAttribute('points')),getStroke(p),p.parentNode.getAttribute('id'),p.toxml()) for p in doc.getElementsByTagName('polyline')]
    #Use minidom to extract polygon strings from input SVG, convert to path strings, add to list
    path_strings += [(polylineStr2pathStr(p.getAttribute('points'))+'z',getStroke(p),p.parentNode.getAttribute('id'),p.toxml()) for p in doc.getElementsByTagName('polygon')]
    #currently choosing to ignore line objects (assuming... all lines are fixes for non-overlapping mergers?)
    ##Use minidom to extract line strings from input SVG, convert to path strings, and add them to list
    #line_strings = [('M' + p.getAttribute('x1') + ' ' +p.getAttribute('y1') + 'L'+p.getAttribute('x2') + ' ' + p.getAttribute('y2'),getStroke(p), p.parentNode.getAttribute('id')) for p in doc.getElementsByTagName('line')]
    doc.unlink()
    opt.basic_output_on.dprint("Done.")

#    #(first attempt to) Check for stray points, if any found, delete them
#    i=0
#    count_popped_points = 0
#    while i < len(path_strings):
#        if path_strings[i][0].count(',')<2:
#            path_strings.pop(i)
#            count_popped_points+=1
#            opt.full_output_on.dprint("Removed a stray point: path_string[%s][0] = %s"%(i,path_strings[i][0]))
#        i +=1
#    opt.basic_output_on.dprint("Removed %s stray points in path_string stage.  Continuing..."%count_popped_points)

    #Convert path_strings to ring objects
    opt.basic_output_on.dprint("Converting path strings to Ring objects.  This could take a minute... ",'nr')
    path2ring_start_time = current_time()
    ring_list = []
    paths_of_unknown_orientation = []
    for i in range(len(path_strings)):
        orig_path = parse_path(path_strings[i][0])
        try: ### DEBUG ONLY (REMOVE ALL OF TRY/EXCEPT)
            orig_path[0]
        except:
            if len(path_strings[i][0].split(','))<3:
                opt.full_output_on.dprint("Found (and skipped) single point path: %s"%path_strings[i][0])
                continue
            else:
                raise

        #fix degenerate segments here
        for index,seg in enumerate(orig_path):
            if abs(seg.start-seg.end) < 1:
                old_end = seg.end
                old_start = seg.start
                opt.full_output_on.dprint("Found degenerate seg in path %s: %s"%(i,seg))
                del orig_path[index]
                if index == len(orig_path): #deleted last path
                    orig_path[-1].end = old_end
                elif index == 0:
                    orig_path[0].start=old_start
                else:
                    orig_path[index].start = orig_path[index-1].end
                opt.full_output_on.dprint("Deleted above degenerate segment and fixed gap.")

        #check for doubled over segments
        nostupidsfound = False
        while not nostupidsfound and len(orig_path)>1:
            for indst in range(len(orig_path)-1):
                if (orig_path[indst] == orig_path[indst+1] or
                    orig_path[indst] == orig_path[indst+1].reversed()):
                    del orig_path[indst+1]
                    opt.warnings_output_on.dprint("[Warning:] "+"stupidsfound"*50)
#                    raise Exception() #you should remove this Exception and everything will run smoothly
            else:
                nostupidsfound = True

        #Now fix the orientation if path is not CCW (w.r.t. center)
        try:
            path_is_ccw = isCCW(orig_path,center)
        except:
            if opt.manually_fix_orientations:
                print("\n[Manually Fix Orientations:] As currently drawn, the "
                      "path starts at the green node/segment and ends at the "
                      "red (if you don't see one of these nodes, it's likely "
                      "cause the path is very short and thus they are on top "
                      "of each other).  Does the path in "
                      "'temporary_4manualOrientation.svg' appear to be drawn "
                      "in a clockwise fashion?")
                if len(orig_path) == 1:
                    disp_paths = [orig_path]
                    disp_path_colors = ['blue']
                elif len(orig_path) == 2:
                    disp_paths = [Path(orig_path[0]),Path(orig_path[1])]
                    disp_path_colors = ['green','red']
                elif len(orig_path) > 2:
                    disp_paths = [Path(orig_path[0]),Path(orig_path[1:-1]),Path(orig_path[-1])]
                    disp_path_colors = ['green','blue','red']
                else:
                    raise Exception("This path is empty... this should never happen.  Tell Andy.")
                for ring in ring_list:
                    disp_paths.append(ring.path)
                    disp_path_colors.append('black')

                nodes = [orig_path[0].start,orig_path[-1].end]+[center]
                node_colors = ['green','red']+[colordict['center']]
                disvg(disp_paths, disp_path_colors, nodes=nodes, node_colors=node_colors, filename='temporary_4manualOrientation.svg')
                path_is_ccw = askUserOrientation() #svg display reverses orientation so a respose of 'yes' means path is actually ccw and thus sets path_is_ccw = True
                if path_is_ccw == 'remove':
                    print("OK, this path will be ignored... moving onto the rest.")
                    continue
#                raise Exception("The manually_fix_orientations feature is not yet setup.  If you need this, ask Andy; it shouldn't take him long.")
            else:
                path_is_ccw = opt.when_orientation_cannot_be_determined_assume_CCW
                paths_of_unknown_orientation.append(path_strings[i])
        if not path_is_ccw:
            path2record = orig_path.reversed()
            opt.full_output_on.dprint("Path %s was not oriented CCW, but is now."%i)
        else:
            path2record = orig_path
        ring_list.append(Ring(path_strings[i][0],path_strings[i][1],path_strings[i][2],rad,path2record,xml=path_strings[i][3]))
        opt.full_output_on.dprint("Ring %s ok"%i)
    if len(paths_of_unknown_orientation)>0:
        from andysmod import ifelse
        fashion = ifelse(opt.when_orientation_cannot_be_determined_assume_CCW,'Counterclockwise','Clockwise')
        ccw_warning = "[Warning:] Unable to determine orientation of %s paths.  This is likely because some paths in this sample are far from being convex.  I assumed that these paths were traced in a %s fashion (to change this assumption, set 'when_orientation_cannot_be_determined_assume_CCW = %s' in options.  If this assumption is false, either the program will crash or the transect will be visibly messed up in the output 'xxx_transects.svg' (where xxx is the input svg's filename sans extension)."%(len(paths_of_unknown_orientation),fashion,not opt.when_orientation_cannot_be_determined_assume_CCW)
        opt.warnings_output_on.dprint(ccw_warning)
    if len(paths_of_unknown_orientation)>1:
        opt.warnings_output_on.dprint("If think you were not consistent tracing in either CCW or CW fashion (or don't get good  output from this file) then set 'manually_fix_orientations = True' in options.")

    #done extracting rings from svg
    opt.basic_output_on.dprint("Done (in %s)."%format_time(current_time()-path2ring_start_time))
    opt.basic_output_on.dprint("Completed extracting rings from SVG. %s rings detected."%len(ring_list))
    return center, ring_list


def palette_check(ring_list):
    opt.basic_output_on.dprint("Palette check running... ",'nr')
    boundary_ring = max([r for r in ring_list if r.isClosed],key=lambda r: r.maxR)
    fixed_count = 0
    for ring in ring_list:
        color = ring.color
        if color not in colordict.values():
            if opt.auto_fix_ring_colors:
                newcolor = closestColor(color,colordict)
                ring.color = newcolor
                opt.colorcheck_output_on.dprint('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC')
                opt.colorcheck_output_on.dprint('WARNING: SVG-creator used a color used that is not in dictionary.')
                opt.colorcheck_output_on.dprint('...changing this ring from %s to %s.' % (color, newcolor))
                opt.colorcheck_output_on.dprint('newcolor in colordict = %s' % (newcolor in colordict))
                fixed_count +=1
            else:
                raise Exception('color used that is not in dictionary')
        if color in [colordict['safe1'],colordict['safe2']]:
            if ring==boundary_ring:
                ring.color = colordict['boundary']
            else:
                ring.color = colordict['complete']
    else:
        opt.basic_output_on.dprint("Palette check passed after fixing %s rings.\n"%fixed_count)

    return ring_list

def closedness_consistency_check(ring_list):
    opt.basic_output_on.dprint("Closedness consistency check running... ",'nr')
    failed_rings = []
    for ring in ring_list:
        ztest = isClosedPathStr(ring.string) #checks for z at end of path string
        atest = isApproxClosedPath(ring.path) #equivalent to isApproxClosedPath(ring.path)
        ctest = ring.color in {colordict['complete'],colordict['boundary']}
        if not ((ztest and atest and ctest) or (not ztest and not atest and not ctest)):
            # import ipdb; ipdb.set_trace()
            failed_rings.append(ring)
            opt.closednessCheck_output_on.dprint("A ring failed the closedness consistency check.")
            opt.closednessCheck_output_on.dprint("Path String: " + str(ring.string))
            opt.closednessCheck_output_on.dprint("color of path: "+str(ring.color))
            opt.closednessCheck_output_on.dprint("closed by z: "+str(ztest))
            opt.closednessCheck_output_on.dprint("approx closed: "+str(atest))
            opt.closednessCheck_output_on.dprint("closed color: "+str(ctest))
            opt.closednessCheck_output_on.dprint("")
            if opt.auto_fix_ring_colors:
                if ztest and not atest: #for ctest or not ctest
                    raise Exception("ztest and not atest")
                elif ztest and atest: #and not ctest
                    ring.color = colordict["complete"]
                    opt.closednessCheck_output_on.dprint("Fixed: changed color to complete color.")
                elif (not ztest) and atest:
                    if ctest:
                        ring.string += 'z'
                        opt.closednessCheck_output_on.dprint("Fixed: appended z to end of string.")
                    else:
                        opt.closednessCheck_output_on.dprint("Maybe this ring is fine as it.  I've fixed nothing.  case: (not ztest) and atest and (not ctest)")
                elif (not ztest) and (not atest): #and ctest
                    ring.color = colordict["incomplete"]
                    opt.closednessCheck_output_on.dprint("Fixed: changed color to incomplete color.")
                else:
                    raise Exception("This case should not logically occur.  There's a bug in my logic.")
    if len(failed_rings) > 0:
        opt.basic_output_on.dprint("Warning:Closedness consistency check failed for %s (out of %s) rings.  "%(len(failed_rings),len(ring_list)),'nr')
        opt.basic_output_on.dprint("I did my best to fix them; set closednessCheck_output_on to True and run again to see the details of my fixes.\n")
    else:
        opt.basic_output_on.dprint("Closedness Consistency check passed with no failed rings.\n")
    return ring_list

def visual_test_of_closed_ring_sort(ring_list):
    visual_test_of_ring_sort([r for r in ring_list if r.isClosed()])
def visual_test_of_ring_sort(ring_list):
    from andysSVGpathTools import svgSlideShow
    from os import path as os_path
    fileloc = os_path.join(opt.output_directory,'debug','ring_sort_slideshow')
    opt.basic_output_on.dprint("Creating SVG slideshow showing ring sorting...",'nr')

    #create svg for each image in slideshow
    pathcolortuplelist = []
    for slide in range(len(ring_list)):
        if slide == 0:
            continue
        elif slide == 1:
            second_to_outer_most_path = next(ring.path for ring in ring_list if ring.sort_index == slide-1)
        else:
            second_to_outer_most_path = outer_most_path
        try: ###Debug (just keep what's inside try)
            outer_most_path = next(ring.path for ring in ring_list if ring.sort_index == slide)
        except:
            bla=1
            raise Exception("bla")
        paths = [ring.path for ring in ring_list if ring.sort_index < slide-1] + [second_to_outer_most_path,outer_most_path]
        colors = ['blue']*(len(paths)-2) + ['red']+['green']
        pathcolortuplelist.append((paths,colors))

    svgSlideShow(pathcolortuplelist,save_directory=fileloc,clear_directory=True,suppressOutput=True)
    opt.basic_output_on.dprint("Done.")
