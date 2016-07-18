def find_ring_areas(sorted_ring_list, center, svgfile):#
    from rings4rings import Ring, IncompleteRing, CompleteRing, CP_BoolSet
    from misc4rings import centerSquare, isCCW
    from andysSVGpathTools import reversePath, disvg
    from andysmod import Radius
    from options4rings import colordict, basic_output_on, showCurrentFilesProgress, outputFolder, create_SVG_showing_area_paths, outputFolder_debug
    from svgpathtools import parse_path
    from os import path as os_path
    #This codeblock creates a one pixel by one pixel square Ring object to act as the core - it is recorded in CP. note: perimeter should be found as a path and treated at a ring already
    csd = centerSquare(center)
    csd_path = parse_path(csd)
    if not isCCW(csd_path,center):
        csd_path = reversePath(csd_path)
    center_square = Ring(csd,colordict['center'], 'not recorded',Radius(center), csd_path)#path_string, color, brooke_tag, center

    #Converts the sorted_ring_list into a CP_Boolset of complete rings each containing their IRs
    completeRing_CPB = CP_BoolSet()
    innerRing = center_square
    innerRing_index = -1
    for ring_index, ring in enumerate(sorted_ring_list):
        if ring.isClosed(): #when next closed ring found create CompleteRing object, then set all inbetween rings to be IRs
            completeRing_CPB.append(CompleteRing(innerRing,ring))
            for inc_ring in sorted_ring_list[innerRing_index+1:ring_index]:
                ir = IncompleteRing(inc_ring)
                ir.set_inner(innerRing)
                ir.set_outer(ring)
                completeRing_CPB.cpUpdate(CompleteRing(ir.innerCR_ring,ir.outerCR_ring,ir))
            innerRing = ring
            innerRing_index = ring_index

    #Check (once again) that the last sort-suggested boundary is closed and correctly colored
    if (not sorted_ring_list[-1].isClosed()) or sorted_ring_list[-1].color != colordict['boundary']:
#        disvg([sorted_ring_list[-1].path],[sorted_ring_list[-1].color],openInBrowser=True)
#        disvg([r.path for r in sorted_ring_list],[r.color for r in sorted_ring_list],openInBrowser=True)
        ###DEBUG Why is this necessary?  Isn't this fixed earlier?
        if sorted_ring_list[-1] == max(sorted_ring_list, key=lambda r: r.maxR):
           sorted_ring_list[-1].color = colordict['boundary']
        else:
            raise Exception("Last ring in sorted sorted_ring_list was not closed... this should be outer perimeter.")

    completeRing_CPB[0].isCore = True #identify the center square created earlier as the core
    basic_output_on.dprint("All complete_ring objects created and all incomple_ring objects created (and stored inside the appropriate complete_ring object).")

    #complete the incomplete rings
    from time import time as current_time
    from andysmod import format_time
    CP_start_time = start_time_ring_completion = current_time()
    for count,cp in enumerate(completeRing_CPB):
        if count:
            CP_start_time = current_time()
        try:
            cp.completeIncompleteRings()
        except:
            from options4rings import outputTroubledCPs
            if outputTroubledCPs:
                from svgpathtools import Line
                from options4rings import colordict
                paths = [cp.inner.path,cp.outer.path]+[ir.ring.path for ir in cp.ir_boolset] + [sorted_ring_list[-1].path]
                path_colors = [cp.inner.color,cp.outer.color]+[ir.ring.color for ir in cp.ir_boolset] + [colordict['boundary']]
                center_line = Line(cp.inner.center-1,cp.inner.center+1)
                svgname = os_path.join(outputFolder_debug,"trouble_"+svgfile)
                disvg(paths,path_colors,lines=[center_line],filename=svgname)
                print "Simplified SVG created containing troublesome section (troublesome incomplete ring colored %s) and saved to:"%colordict['safe1']
                print svgname
            raise

        mes = "%s/%s complete rings finished. This CP = %s | Total ET = %s"%(count+1,
               len(completeRing_CPB),
                format_time(current_time()-CP_start_time),
                format_time(current_time()-start_time_ring_completion))
        showCurrentFilesProgress.dprint(mes)

    outputFile = os_path.join(outputFolder,svgfile+'_completeRing_info.csv')
    with open(outputFile,"wt") as out_file:
        out_file.write("complete ring index, type, # of IRs contained, minR, maxR, aveR, area, area Ignoring IRs\n")
        cp_index = 0
        for cp in completeRing_CPB:
            cp_index += 1
            out_file.write(cp.info(cp_index,colordict) + '\n')

    #Create SVG showing ring sorting
    if create_SVG_showing_area_paths:
        basic_output_on.dprint("Attempting to create SVG showing completed paths used for area computation...",'nr')
        svgpaths = []
        svgcolors = []
        for cp in completeRing_CPB:
            svgpaths.append(cp.inner.path)
            svgcolors.append(cp.inner.color)
            for ir in cp.ir_boolset:
                svgpaths.append(ir.completed_path)
                svgcolors.append(ir.ring.color)
            if cp.outer.color == colordict['boundary']:
                svgpaths.append(cp.outer.path)
                svgcolors.append(cp.outer.color)

        svgname = os_path.join(outputFolder_debug,svgfile[0:len(svgfile)-4]+"_area_paths"+".svg")
        disvg(svgpaths,svgcolors,filename=svgname)
        basic_output_on.dprint("Done.")