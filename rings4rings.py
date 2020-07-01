from misc4rings import (Theta_Tstar, normalLineAtT_toInner_intersects_withOuter,
                        isDegenerateSegment, isCCW, areaEnclosed,
                        isApproxClosedPath, pathXpathIntersections,
                        remove_degenerate_segments)
import options4rings as opt
from andysmod import boolset
from andysSVGpathTools import (path2str, printPath, pathT2tseg, cropPath,
                               reversePath, cubPoints, minRadius, maxRadius,
                               trimSeg, segt2PathT, reverseSeg,
                               closestPointInPath, concatPaths,
                               lineXlineIntersections)
from svgpathtools import parse_path, Path, Line, CubicBezier, disvg
from copy import deepcopy as copyobject

from operator import itemgetter
def sortby(x, k):
    return sorted(x, key=itemgetter(k))


class Ring(object):
    def __init__(self, path_string, color, brook_tag, rad, path, xml=None):
        self.string = str(path_string)
        self.xml = xml  # original xml string the this ring came from in input svg
        self.center = rad.origin
        self.rad = rad
        self.color = color
        self.brook_tag = brook_tag
        self._path = path
        self.path_before_removing_intersections = None
        if opt.save_path_length_in_pickle:
            #Calculate lengths of segments in path so this info gets stored in
            # pickle file (time consuming)
            self.path._calc_lengths()
        self.minR = minRadius(self.center,self.path)
        self.maxR = maxRadius(self.center,self.path)
        self.isAbove = set()
        self.isBelow = set()
        self.sort_index = None  # flattened sort index (unique numerical)
        self.psort_index = None  # partial sort index (alphanumeric string)
        self.wasClosed = None # records the closure before removing intersections (shouldn't ever differ from isClosed())
        self.svgname = None
        self.nL2bdry_a = None # Path(Line(curve_pt,bdry_pt))
        self.nL2bdry_b = None # Path(Line(curve_pt,bdry_pt))
        self.pathAroundBdry = None # closed path given by path+l2b1+bdry_path+l2b0

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, new_path):
        self._path = new_path
        self.minR = minRadius(self.center, self._path)
        self.maxR = maxRadius(self.center, self._path)
        self.string = path2str(new_path)


    def findLines2Bdry(self,bdry_rectangle):
        ((T_a,S2_a),(T_b,S2_b)) = find_intersection_of_rectangle_with_paths_outward_pointing_normal_line_bundle(self.path,bdry_rectangle)
        self.nL2bdry_a = Path(Line(self.path.point(T_a),bdry_rectangle.point(S2_a)))
        self.nL2bdry_b = Path(Line(self.path.point(T_b),bdry_rectangle.point(S2_b)))
    def path_around_bdry(self,bdry_path):
        if not self.pathAroundBdry:
            raise Exception("The following normalLines don't work... the don't even attach to endpoints")
            nL0,seg0,t0 = normalLineAtT_toInner_intersects_withOuter(self.nL2bdry_a,self.path,bdry_path,self.center)
            nL1,seg1,t1 = normalLineAtT_toInner_intersects_withOuter(self.nL2bdry_b,self.path,bdry_path,self.center)
            l2b0 = Path(reverseSeg(nL0))
            l2b1 = Path(nL1)
            T0 = segt2PathT(bdry_path,seg0,t0)
            T1 = segt2PathT(bdry_path,seg1,t1)
            inters = lineXlineIntersections(nL0,nL1)
            if not inters:
                bdry_part = reversePath(cropPath(bdry_path,T0,T1))
            elif len(inters)==1:
                bdry_part = reversePath(cropPath(bdry_path,T1,T0))
            else:
                raise Exception("This case should never be reached.")
            self.pathAroundBdry = concatPaths([self.path,l2b1,bdry_part,l2b0])
        return self.pathAroundBdry

    def record_wasClosed(self):
        self.wasClosed = self.isApproxClosedRing()
    def isApproxClosedRing(self):
        return abs(self.path[-1].end - self.path[0].start) < opt.tol_isApproxClosedPath
    def isClosed(self):
        return self.path[-1].end == self.path[0].start
    def endpt(self):
        return self.path[-1].end
    def startpt(self):
        return self.path[0].start
    def fixClosure(self):
        #Remove degenerate segments
        for i,seg in enumerate(self.path):
            if seg.start==seg.end:
                del self.path[i]
        #Close the ring
        if (abs(self.endpt() - self.startpt()) < opt.tol_isApproxClosedPath and
                    self.path.length() > opt.appropriate_ring_length_minimum):
            self.path[-1].end = self.path[0].start

#    def __repr__(self):
#        return '<Ring object of color = %s, Brook_tag = %s, minR = %s, maxR = %s>' %(self.color,self.brooke_tag,self.minR, self.maxR)
    def updatePath(self, new_path):
        if self.path_before_removing_intersections is None:
            self.path_before_removing_intersections = self.path
        self.path = new_path


    def __eq__(self, other):
        if not isinstance(other, Ring):
            return NotImplemented
        if self.path != other.path or self.string!=other.string:
            return False
        return True

    def __ne__(self, other):
        if not isinstance(other, Ring):
            return NotImplemented
        return not self == other

    def point(self, pos):
        return self.path.point(pos)

    def parseCCW(self):
        orig_path = parse_path(self.string)
        #fix degenerate segments here
        for i,seg in enumerate(orig_path):
            if abs(seg.start-seg.end) < 1:
                del orig_path[i]
                orig_path[i].start = orig_path[i-1].end
        if isCCW(orig_path,self.center):
            return orig_path
        else:
            return reversePath(orig_path)

    def aveR(self):
#        return aveRadius_path(self.path,self.center)
        return "Not Implimented"

    def area(self):
        if not self.isClosed():
            raise Exception("Area of ring object can can only be measured with this function if it is a closed (complete) ring.  You must make it into an incomplete ring object and give a completed_path.")
        return areaEnclosed(self.path)

#    def info(self, cp_index):
#        ###### "complete ring index, complete?, inner BrookID, outer BrookID, inner color, outer color, area, area Ignoring IRs, averageRadius, minR, maxR, IRs contained"
#        return str(cp_index) + "," + "True" + ".... sorry not implimented yet"



class IncompleteRing(object):
    def __init__(self, ring):
        self.ring = ring
        self.innerCR_ring = None
        self.outerCR_ring = None
        self.completed_path = Path()
        self.overlap0 = False #This is related to a deprecated piece of code and must be False.
        self.overlap1 = False #This is related to a deprecated piece of code and must be False.
        self.corrected_start = None #set in case of overlap (point, seg,t) where seg is a segment in self and seg(t)=point
        self.corrected_end = None #set in case of overlap (point, seg,t) where seg is a segment in self and seg(t)=point
        self.ir_start = self.ring.point(0)
        self.ir_end = self.ring.point(1)
        self.up_ladders = []
        self.down_ladder0 = None #(irORcr0,T0) ~ startpoint down-ladder on this ir and (and T-val on connecting ring it connects at - irORcr0 can be incompleteRing object or completeRing object)
        self.down_ladder1 = None
        self.transect0fails = [] #records IRs are "below" self, but failed to provide a transect to self.ir_start
        self.transect1fails = [] #records IRs are "below" self, but failed to provide a transect to self.ir_end
        self.transect0found = False
        self.transect1found = False
        self.isCore = False
        self.ORring = self.ring

#    def __repr__(self):
#        return '<IncompleteRing based on ring = %s>' %self.ring
    def __eq__(self, other):
        if not isinstance(other, IncompleteRing):
            return NotImplemented
        if self.ring != other.ring:
            return False
        return True
    def __ne__(self, other):
        if not isinstance(other, CompleteRing):
            return NotImplemented
        return not self == other

    def set_inner(self, ring):
        self.innerCR_ring = ring
    def set_outer(self, ring):
        self.outerCR_ring = ring

    def sortUpLadders(self):
        self.up_ladders = sortby(self.up_ladders,1)
        self.up_ladders.reverse()

# this as my newer cleaned up version, but I broke it i think (7-19-16)
    # def addSegsToCP(self, segs, tol_closure=opt.tol_isApproxClosedPath):
    #     """input a list of segments to append to self.completed_path
    #     this function will stop adding segs if a seg endpoint is near the
    #     completed_path startpoint"""
    #     if len(segs)==0:
    #         raise Exception("No segs given to insert")
    #
    #     # Iterate through segs to check if segments join together nicely
    #     # and (fix them if need be and) append them to completed_path
    #     for seg in segs:
    #         # This block checks if cp is (nearly) closed.
    #         # If so, closes it with a Line, and returns the fcn
    #         if len(self.completed_path)!=0:
    #             cp_start, cp_end = self.completed_path[0].start, self.completed_path[-1].end
    #             if abs(cp_start - cp_end) < tol_closure:
    #                 if cp_start==cp_end:
    #                 # then do nothing else and return
    #                     return
    #                 else:
    #                 # then close completed_path with a line and return
    #                     self.completed_path.append(Line(cp_start, cp_end))
    #                     return
    #
    #             elif seg.start != self.completed_path[-1].end:
    #                 # then seg does not start at the end of completed_path,
    #                 # fix it then add it on
    #                 current_endpoint = self.completed_path[-1].end
    #                 if abs(seg.start - current_endpoint) <  tol_closure:
    #                     # then seg is slightly off from current end of
    #                     # completed_path, fix seg and insert it into
    #                     # completed_path
    #                     if isinstance(seg, CubicBezier):
    #                         P0, P1, P2, P3 = seg.bpoints()
    #                         newseg = CubicBezier(current_endpoint, P1, P2, P3)
    #                     elif isinstance(seg, Line):
    #                         newseg = Line(current_endpoint, seg.end)
    #                     else:
    #                         raise Exception('Path segment is neither Line '
    #                                         'object nor CubicBezier object.')
    #                     self.completed_path.insert(len(self.completed_path), newseg)
    #                 else:
    #                     raise Exception("Segment being added to path does not "
    #                                     "start at path endpoint.")
    #             else:
    #                 # then seg does not need to be fixed, so go ahead and insert it
    #                 self.completed_path.insert(len(self.completed_path), seg)

    def addSegsToCP(self, segs, tol_closure=opt.tol_isApproxClosedPath):
    #input a list of segments to append to self.completed_path
    #this function will stop adding segs if a seg endpoint is near the completed_path startpoint
        if len(segs)==0:
            raise Exception("No segs given to insert")

        #Iterate through segs to check if segments join together nicely
        #and (fix them if need be and) append them to completed_path
        for seg in segs:
            #This block checks if cp is (nearly) closed.
            #If so, closes it with a Line, and returns the fcn
            if len(self.completed_path)!=0:
                cp_start, cp_end = self.completed_path[0].start, self.completed_path[-1].end
                if abs(cp_start - cp_end) < tol_closure:
                    if cp_start==cp_end:
                    #then do nothing else and return
                        return
                    else:
                    #then close completed_path with a line and return
                        self.completed_path.append(Line(cp_start,cp_end))
                        return

            if len(self.completed_path)!=0 and seg.start != self.completed_path[-1].end:
            #then seg does not start at the end of completed_path, fix it then add it on
                current_endpoint = self.completed_path[-1].end
                if abs(seg.start - current_endpoint) <  tol_closure:
                #then seg is slightly off from current end of completed_path, fix seg and insert it into completed_path
                    if isinstance(seg,CubicBezier):
                        P0,P1,P2,P3 = cubPoints(seg)
                        newseg = CubicBezier(current_endpoint,P1,P2,P3)
                    elif isinstance(seg,Line):
                        newseg = Line(current_endpoint,seg.end)
                    else:
                        raise Exception('Path segment is neither Line object nor CubicBezier object.')
                    self.completed_path.insert(len(self.completed_path),newseg)
                else:
                    raise Exception("Segment being added to path does not start at path endpoint.")
            else:
            #then seg does not need to be fixed, so go ahead and insert it
                self.completed_path.insert(len(self.completed_path),seg)

    def addConnectingPathToCP(self, connecting_path, seg0, t0, seg1, t1):
        # first find orientation by checking whether t0 is closer to start or end.
        T0, T1 = segt2PathT(connecting_path, seg0, t0), segt2PathT(connecting_path, seg1, t1)
        i0, i1 = connecting_path.index(seg0), connecting_path.index(seg1)
        first_seg = reverseSeg(trimSeg(seg1, 0, t1))
        last_seg = reverseSeg(trimSeg(seg0, t0, 1))

        if T0 > T1:  # discontinuity between intersection points
            if isApproxClosedPath(connecting_path):
                middle_segs = [reverseSeg(connecting_path[i1-i]) for i in range(1, (i1-i0) % len(connecting_path))]
            else:
                raise Exception("ir jumps another ir's gap.  This case is not "
                                "implimented yet")
        elif T0 < T1:  # discontinuity NOT between intersection points
            middle_segs = [reverseSeg(connecting_path[i1+i0-i]) for i in range(i0 + 1, i1)]
        else:
            raise Exception("T0=T1, this means there's a bug in either "
                            "pathXpathIntersections fcn or "
                            "trimAndAddTransectsBeforeCompletion fcn")

        # first seg
        if isDegenerateSegment(first_seg):
            tmpSeg = copyobject(middle_segs.pop(0))
            tmpSeg.start = first_seg.start
            first_seg = tmpSeg
        if first_seg.end == self.completed_path[0].start:
            self.completed_path.insert(0,first_seg)
        else:
            printPath(first_seg)
            printPath(last_seg)
            printPath(connecting_path)
            raise Exception("first_seg is set up wrongly")

        # middle segs
        self.addSegsToCP(middle_segs)

        # last seg
        if isDegenerateSegment(last_seg):
            middle_segs[-1].end = last_seg.end
        else:
            self.addSegsToCP([last_seg])

    def trimAndAddTransectsBeforeCompletion(self):

        # Retrieve transect endpoints if necessary
        (irORcr0, T0), (irORcr1, T1) = self.down_ladder0, self.down_ladder1
        tr0_start_pt = irORcr0.ORring.point(T0)
        tr1_end_pt = irORcr1.ORring.point(T1)

        if not self.overlap0:
            # then no overlap at start, add transect0 to beginning of
            # connected path (before the ir itself)
            i0 = -1
            startSeg = Line(tr0_start_pt, self.ir_start)
        else:
            # overlap at start, trim the first seg in the ir (don't connect
            # anything, just trim)
            i0 = self.ring.path.index(self.corrected_start[1])
            startSeg = trimSeg(self.corrected_start[1], self.corrected_start[2],1)
        if not self.overlap1:
            # then no overlap at end to add transect1 to connected path
            # (append to end of the ir)
            i1 = len(self.ring.path)
            endSeg = Line(self.ir_end, tr1_end_pt)
        else:
            # overlap at end, trim the last seg in the ir (don't connect
            # anything, just trim)
            i1 = self.ring.path.index(self.corrected_end[1])
            endSeg = trimSeg(self.corrected_end[1], 0, self.corrected_end[2])

        # first seg
        if isDegenerateSegment(startSeg):
            tmpSeg = copyobject(self.ring.path[i0 + 1])
            tmpSeg.start = startSeg.start
            startSeg = tmpSeg
            i0 += 1
            self.addSegsToCP([startSeg])
        else:
            self.addSegsToCP([startSeg])

        # middle segs
        if i0 + 1 != i1:
            self.addSegsToCP([self.ring.path[i] for i in range(i0+1, i1)])

        # last seg
        if isDegenerateSegment(endSeg):
            self.completed_path[-1].end = endSeg.end
        else:
            self.addSegsToCP([endSeg])

    def irpoint(self, pos):
        return self.ring.point(pos)

    def area(self):
        if not isinstance(self.completed_path,Path):
            return "Fail"
        if (self.completed_path is None or not isApproxClosedPath(self.completed_path)):
            # raise Exception("completed_path not complete.  Distance between start and end: %s"%abs(self.completed_path.point(0) - self.completed_path.point(1)))
            return "Fail"
        return areaEnclosed(self.completed_path)

    def type(self, colordict):
        for (key, val) in colordict.items():
            if self.ring.color == val:
                return key
        else:
            raise Exception("Incomplete Ring color not in colordict... you shouldn't have gotten this far.  Bug detected.")

#    def info(self,cp_index):
#    ###### "complete ring index, complete?, inner BrookID, outer BrookID, inner color, outer color, area, area Ignoring IRs, averageRadius, minR, maxR, IRs contained"
#        return str(cp_index) + "," + "Incomplete"+"," + "N/A" + ", " + self.ring.brook_tag + "," + "N/A" + ", " + self.ring.color +"," + str(self.area()) +", "+ "N/A"+","+str(self.ring.aveR())+","+str(self.ring.minR)+","+str(self.ring.maxR)+","+"N/A"

    def info(self, cp_index, colordict):
        ###### "complete ring index, type, # of IRs contained, minR, maxR, aveR, area, area Ignoring IRs"
        return str(cp_index)+","+self.type(colordict)+","+"N/A"+","+str(self.ring.minR)+","+ str(self.ring.maxR)+","+str(self.ring.aveR())+","+str(self.area())+","+"N/A"

    def followPathBackwards2LadderAndUpDown(self, irORcr, T0):
        """irORcr is the path being followed, self is the IR to be completed
        returns (traveled_path,irORcr_new,t_new) made from the part of irORcr's
        path before T0 (and after ladder) plus the line from ladder (the first
        one that is encountered)"""
        rds = remove_degenerate_segments
        irORcr_path = irORcr.ORring.path
        thetaprekey = Theta_Tstar(T0)
        thetakey = lambda lad: thetaprekey.distfcn(lad[1])
        sorted_upLadders = sorted(irORcr.up_ladders, key=thetakey)

        if isinstance(irORcr, CompleteRing):
            ir_new, T = sorted_upLadders[0]
            if T != T0:
                reversed_path_followed = reversePath(cropPath(irORcr_path, T, T0))
            else:  # this happens when up and down ladder are at same location
                reversed_path_followed = Path()

            # add the ladder to reversed_path_followed
            if (irORcr, T) == ir_new.down_ladder0:
                if not ir_new.overlap0:
                    ladder = Line(irORcr_path.point(T), ir_new.irpoint(0))
                    reversed_path_followed.append(ladder)
                    T_ir_new = 0
                else:
                    T_ir_new = segt2PathT(ir_new.ring.path,
                                          ir_new.corrected_start[1],
                                          ir_new.corrected_start[2])
            elif (irORcr, T) == ir_new.down_ladder1:
                if not ir_new.overlap1:
                    ladder = Line(irORcr_path.point(T), ir_new.irpoint(1))
                    reversed_path_followed.append(ladder)
                    T_ir_new = 1
                else:
                    T_ir_new = segt2PathT(ir_new.ring.path,
                                          ir_new.corrected_end[1],
                                          ir_new.corrected_end[2])
            else:
                raise Exception("this case shouldn't be reached, mistake in "
                                "logic or didn't set downladder somewhere.")
            return rds(reversed_path_followed), ir_new, T_ir_new

        else:  # current ring to follow to ladder is incomplete ring
            irORcr_path = irORcr.ring.path
            for ir_new, T in sorted_upLadders:
                if T < T0:  # Note: always following path backwards
                    reversed_path_followed = irORcr_path.cropped(T, T0).reversed()
                    if (irORcr, T) == ir_new.down_ladder0:

                        if not ir_new.overlap0:
                            ladder = Line(irORcr_path.point(T), ir_new.irpoint(0))
                            reversed_path_followed.append(ladder)
                            T_ir_new = 0
                        else:
                            T_ir_new = segt2PathT(ir_new.ring.path,
                                                  ir_new.corrected_start[1],
                                                  ir_new.corrected_start[2])
                    elif (irORcr, T) == ir_new.down_ladder1:
                        if not ir_new.overlap1:
                            ladder = Line(irORcr_path.point(T), ir_new.irpoint(1))
                            reversed_path_followed.append(ladder)
                            T_ir_new = 1
                        else:
                            T_ir_new = segt2PathT(ir_new.ring.path,
                                                  ir_new.corrected_end[1],
                                                  ir_new.corrected_end[2])
                    else:
                        tmp_mes = ("this case shouldn't be reached, mistake "
                                   "in logic or didn't set downladder "
                                   "somewhere.")
                        raise Exception(tmp_mes)

                    return rds(reversed_path_followed), ir_new, T_ir_new

            # none of the upladder were between 0 and T0,
            # so use downladder at 0
            else:
                (irORcr_new, T_new) = irORcr.down_ladder0
                irORcr_new_path = irORcr_new.ORring.path

                ###Should T0==0 ever?
                if T0 != 0:
                    reversed_path_followed = irORcr.ring.path.cropped(0, T0).reversed()
                else:
                    reversed_path_followed = Path()

                if irORcr.overlap0 == False:
                    ladder = Line(irORcr_path.point(0), irORcr_new_path.point(T_new))
                    reversed_path_followed.append(ladder)
                return rds(reversed_path_followed), irORcr_new, T_new

    def findMiddleOfConnectingPath(self):
        #creates path starting of end of down_ladder1, go to nearest (with t> t0) up-ladder or if no up-ladder then down_ladder at end and repeat until getting to bottom of down_ladder0
        maxIts = 1000 #### Tolerance
        traveled_path = Path()
        iters = 0
        (irORcr_new,T_new) = self.down_ladder1
        doneyet = False
        while iters < maxIts and not doneyet:
            iters =iters+ 1

#            ##DEBUG sd;fjadsfljkjl;
#            if self.ring.path[0].start == (260.778+153.954j):
#                from misc4rings import dis
#                from svgpathtools import Line
#                p2d=[self.completed_path,
#                     self.down_ladder0[0].ORring.path,
#                     self.down_ladder1[0].ORring.path]
#                clrs = ['blue','green','red']
#                if iters>1:
#                    p2d.append(Path(*traveled_path))
#                    clrs.append('black')
#                lad0a = self.down_ladder0[0].ORring.path.point(self.down_ladder0[1])
#                lad0b = self.ORring.path[0].start
#                lad0 = Line(lad0a,lad0b)
#                lad1a = self.ORring.path[-1].end
#                lad1b = self.down_ladder1[0].ORring.path.point(self.down_ladder1[1])
#                lad1 = Line(lad1a,lad1b)
#                dis(p2d,clrs,lines=[lad0,lad1])
#                print abs(lad0.start-lad0.end)
#                print abs(lad1.start-lad1.end)
#                bla=1
#            ##end of DEBUG sd;fjadsfljkjl;
            
            traveled_path_part, irORcr_new, T_new = self.followPathBackwards2LadderAndUpDown(irORcr_new, T_new)

            for seg in traveled_path_part:
                traveled_path.append(seg)

            if irORcr_new == self:
                return traveled_path
#            if irORcr_new == self.down_ladder0[0]:
#                doneyet = True
#                irORcr_new_path = irORcr_new.ORring.path
#                if T_new != self.down_ladder0[1]:
#                    for seg in reversePath(cropPath(irORcr_new_path,self.down_ladder0[1],T_new)):
#                        traveled_path.append(seg)
#                break
            if (irORcr_new, T_new) == self.down_ladder0:
                return traveled_path

            if iters >= maxIts-1:
                raise Exception("findMiddleOfConnectingPath reached maxIts")
        return traveled_path

    def hardComplete(self, tol_closure=opt.tol_isApproxClosedPath):
        self.trimAndAddTransectsBeforeCompletion()
        meatOf_connecting_path = self.findMiddleOfConnectingPath()  ###this is a Path object
        self.addSegsToCP(meatOf_connecting_path)
        cp_start,cp_end = self.completed_path[0].start, self.completed_path[-1].end

        #check newly finished connecting_path is closed
        if abs(cp_start - cp_end) >= tol_closure:
            raise Exception("Connecting path should be finished but is not closed.")

        #test for weird .point() bug where .point(1) != end
        if (abs(cp_start - self.completed_path.point(0)) >= tol_closure or
            abs(cp_end - self.completed_path.point(1)) >= tol_closure):
            self.completed_path = parse_path(path2str(self.completed_path))
            raise Exception("weird .point() bug where .point(1) != end... I just added this check in on 3-5-15, so maybe if this happens it doesn't even matter.  Try removing this code-block... or change svgpathtools.Path.point() method to return one or the other.")

    def findTransect2endpointFromInnerPath_normal(self,irORcr_innerPath,innerPath,T_range,Tpf,endBin):
        #Tpf: If The T0 transect intersects self and the T1 does not, then Tpf should be True, otherwise it should be false.
        #Note: If this endpoint's transect is already found, then this function returns (False,False,False,False)
        #Returns: (irORcr,nL,seg_irORcr,t_irORcr) where irORcr is the inner path that the transect, nL, leaves from and seg_irORcr and t_irORcr correspond to innerPath and nL points from seg_irORcr.point(t_irORcr) to the desired endpoint
        #Note: irORcr will differ from irORcr_innerPath in the case where irORcr_innerPath admits a transect but this transect intersects with a (less-inner) previously failed path.  The less-inner path is then output.
        (T0,T1) = T_range
        if T1<T0:
            if T1==0:
                T1=1
            else:
                Exception("I don't think T_range should ever have T0>T1.  Check over findTransect2endpointsFromInnerPath_normal to see if this is acceptable.")

        if endBin == 0 and self.transect0found:
            return False, False, False, False
        elif endBin == 1 and self.transect1found:
             return False, False, False, False
        elif endBin not in {0,1}:
            raise Exception("endBin not a binary - so there is a typo somewhere when calling this fcn")

        if irORcr_innerPath.isCore:
            return irORcr_innerPath, Line(irORcr_innerPath.inner.path.point(0), self.irpoint(endBin)), irORcr_innerPath.inner.path[0], 0  # make transect from center (actually path.point(0)) to the endpoint

        maxIts = 100 ##### tolerance
        its = 0
        while (abs(innerPath.point(T0) - innerPath.point(T1)) >= opt.tol_isApproxClosedPath
               and its <= maxIts):
            its += 1
            T = float((T0+T1))/2
            center = self.ring.center
            nL = normalLineAtT_toInner_intersects_withOuter(T,innerPath,self.ring.path,center)[0] #check if transect from innerPath.point(T) intersects with outerPath
            if Tpf: #p x f
                if nL != False:  #p p f
                    T0 = T
                else: #p f f
                    T1 = T
            else: #f x p
                if nL != False: #f p p
                    T1 = T
                else: #f f p
                    T0 = T
#            ###DEBUG asdfkljhjdkdjjjdkkk
#            if self.ORring.point(0)==(296.238+285.506j):
#                from misc4rings import dis
#                print "endBin=%s\nT0=%s\nT=%s\nT1=%s\n"%(endBin,T0,T,T1)
#                if isNear(innerPath.point(T0),innerPath.point(T1)):
#                    print "Exit Criterion Met!!!"
#                    if nL==False:
#                        nLtmp = normalLineAtT_toInner_intersects_withOuter(T,innerPath,self.ring.path,center,'debug')[0]
#                    else:
#                        nLtmp = nL
#                    dis([innerPath,self.ring.path,Path(nLtmp)],['green','red','blue'],nodes=[center,innerPath.point(T0),innerPath.point(T1)],node_colors=['blue','green','red'])
#                    bla=1
#            ###end of DEBUG asdfkljhjdkdjjjdkkk
            if its>=maxIts:
                raise Exception("while loop for finding transect by bisection reached maxIts without terminating")
        if nL != False: #x p x
            t_inner, seg_inner = pathT2tseg(innerPath, T)
        else: #x f x
            if Tpf: #p f f
                nL = normalLineAtT_toInner_intersects_withOuter(T0,innerPath,self.ring.path,center)[0]
                (t_inner,seg_inner) = pathT2tseg(innerPath,T0)
            else: #f f p
                nL = normalLineAtT_toInner_intersects_withOuter(T1,innerPath,self.ring.path,center)[0]
                (t_inner,seg_inner) = pathT2tseg(innerPath,T1)
        transect_info = (irORcr_innerPath,nL,seg_inner,t_inner)

        ###Important Note: check that transect does not go through any other rings while headed to its target endpoint (Andy has note explaining this "7th case")
        ###If this intersection does happen... just cut off the line at the intersection point - this leads to transect not being normal to the ring it emanates from.
        if endBin == 0:
            failed_IRs_2check = self.transect0fails
        else:
            failed_IRs_2check = self.transect1fails
        keyfcn = lambda x: x.ORring.sort_index
        failed_IRs_2check = sorted(failed_IRs_2check,key=keyfcn)
        tr_line = transect_info[1]
        exclusions = [] #used to check the line from closest_pt to endpoint doesn't intersect
        num_passed = 0
        run_again = True
        while run_again:
            run_again = False
            for idx,fIR in enumerate(failed_IRs_2check):
                num_passed +=1
                if idx in exclusions:
                    continue
                intersectionList = pathXpathIntersections(Path(tr_line),fIR.ring.path)
                if len(intersectionList) == 0:
                    continue
                else:
                    if len(intersectionList) > 1:
                        print("Warning: Transect-FailedPath intersection check returned multiple intersections.  This is possible, but should be very rare.")
    #                (seg_tl, seg_fIR, t_tl, t_fIR) = intersectionList[0]
                    t_fIR,seg_fIR = closestPointInPath(self.ORring.point(endBin),fIR.ring.path)[1:3]
                    fIR_closest_pt = seg_fIR.point(t_fIR)
                    if endBin:
                        new_nonNormal_transect = Line(self.ring.path[-1].end,fIR_closest_pt)
                    else:
                        new_nonNormal_transect = Line(fIR_closest_pt,self.ring.path[0].start)
                    transect_info = (fIR,new_nonNormal_transect,seg_fIR,t_fIR)
                    exclusions += range(idx+1)
                    run_again = True
                    break
        return transect_info

    def findTransects2endpointsFromInnerPath_normal(self,irORcr_innerPath,innerPath):
        """Finds transects to both endpoints (not just that specified by
        endBin - see outdated description below)
        Note: This fcn will attempt to find transects for endpoints where the
        transects have not been found and will return (False, False, False)
        for those that have.
        Returns: (irORcr,nL,seg_irORcr,t_irORcr) where irORcr is the inner
        path that the transect, nL, leaves from and seg_irORcr and t_irORcr
        correspond to innerPath and nL points from seg_irORcr.point(t_irORcr)
        to the desired endpoint.
        Note: irORcr will differ from irORcr_innerPath in the case where
        irORcr_innerPath admits a transect but this transect intersects with a
        (less-inner) previously failed path.  The less-inner path is then
        output."""
        #Outdated Instructions
        #This fcn is meant to find the transect line from (and normal to) the
        # inner path that goes through OuterPt.  It does this using the
        # bisection method.
        #INPUT: innerPath and outerPath are Path objects, center is a point
        # representing the core, endBin specifies which end point in outerPath
        # we hope to find the transect headed towards (so must be a 0 or a 1)
        #OUTPUT: Returns (transect_Line,inner_seg,inner_t) where normal_line
        # is the transverse Line object normal to innerPath intersecting
        # outerPath at outerPt or, if such a line does not exist, returns
        # (False,False,False,False)
        # inner_seg is the segment of innerPath that this normal transect line
        # begins at, s.t. seg.point(inner_t) = transect_Line.point(0)

        outerPath = self.ring.path
        center = self.ring.center

        tol_numberDetectionLines_transectLine_normal = 20 ##### tolerance
        N = tol_numberDetectionLines_transectLine_normal

#        if self.transect0found and not self.transect1found:
#            return (False,False,False,False), self.findTransect2endpointFromInnerPath_normal(innerPath)
#        if not self.transect0found and self.transect1found:
#            return self.findTransect2endpoint0FromInnerPath_normal(innerPath), (False,False,False,False)
#        if self.transect0found and self.transect1found:
#            raise Exception("Both transects already found... this is a bug.")

        # For a visual explanation of the following code block and the six
        # cases, see Andy's "Gap Analysis of Transects to Endpoints"

        # check if transect from innerPath.point(0) intersect with outerPath
        nL_from0, seg_from0, t_from0 = normalLineAtT_toInner_intersects_withOuter(0, innerPath, outerPath, center)
        if isApproxClosedPath(innerPath):
            (nL_from1,seg_from1,t_from1) = (nL_from0,seg_from0,t_from0)
        else:
            #check if transect from innerPath.point(1) intersect with outerPath
            nL_from1, seg_from1, t_from1 = normalLineAtT_toInner_intersects_withOuter(1, innerPath, outerPath, center)
        #Case: TF
        if nL_from0 and not nL_from1:
            return (False,False,False,False), self.findTransect2endpointFromInnerPath_normal(irORcr_innerPath,innerPath,(0,1),True,1)
        #Case: FT
        if (not nL_from0) and nL_from1:
            return self.findTransect2endpointFromInnerPath_normal(irORcr_innerPath,innerPath,(0,1),False,0), (False,False,False,False)

        # determine All, None, or Some (see notes in notebook on this agorithm
        # for explanation)
        max_pass_Tk = 0
        min_pass_Tk = 1
        max_fail_Tk = 0
        min_fail_Tk = 1
        somePass = False
        someFail = False
        dT = float(1)/(N-1)
        for k in range(1,N-1):
            Tk = k*dT
            nLk, outer_segk, outer_tk = normalLineAtT_toInner_intersects_withOuter(Tk, innerPath, outerPath, center)
            if nLk != False:
                somePass = True
                if Tk > max_pass_Tk:
                    max_pass_Tk = Tk
#                    max_pass = (nLk,outer_segk,outer_tk)
                if Tk < min_pass_Tk:
                    min_pass_Tk = Tk
#                    min_pass = (nLk,outer_segk,outer_tk)
            else:
                someFail = True
                if Tk > max_fail_Tk:
                    max_fail_Tk = Tk
#                    max_fail = (nLk,outer_segk,outer_tk)
                if Tk < min_fail_Tk:
                    min_fail_Tk = Tk
#                    min_fail = (nLk,outer_segk,outer_tk)

        if somePass and someFail:
            #Case: TT & only some pass    [note: TT & some iff TT & T0>T1]
            if nL_from0 and nL_from1:
                Trange0 = (max_fail_Tk, (max_fail_Tk + dT)%1)
                Tpf0 = False
                Trange1 = ((min_fail_Tk - dT)%1, min_fail_Tk)
                Tpf1 = True
            #Case: FF & only some pass
            elif (not nL_from0) and (not nL_from1):
                Trange0 = ((min_pass_Tk - dT)%1, min_pass_Tk)
                Tpf0 = False
                Trange1 = (max_pass_Tk, (max_pass_Tk + dT)%1)
                Tpf1 = True
            for Ttestindex,T2test in enumerate(Trange0 + Trange1): #debugging only
                if T2test>1 or T2test < 0:
                    print(Ttestindex)
                    print(T2test)
                    raise Exception()
            args = irORcr_innerPath, innerPath, Trange0, Tpf0, 0
            tmp1 = self.findTransect2endpointFromInnerPath_normal(*args)
            args = irORcr_innerPath, innerPath, Trange1, Tpf1, 1
            tmp2 = self.findTransect2endpointFromInnerPath_normal(*args)
            return tmp1, tmp2
        #Cases: (TT & all) or (FF & none)    [note: TT & all iff TT & T0<T1]
        else:
            return (False, False, False, False), (False, False, False, False)


class CompleteRing(object):  # this must remain fast to initialize
    def __init__(self, innerRing, outerRing, *internalRings):
        self.inner = innerRing
        self.outer = outerRing
        self.ir_boolset = boolset(internalRings)
        self.up_ladders = []
        self.isCore = False
        self.ORring = self.inner

    def sortUpLadders(self):
        self.up_ladders = sortby(self.up_ladders,1)
        self.up_ladders.reverse()

    def sortIRs(self):
        self.ir_boolset = sorted(self.ir_boolset, key = lambda ir: ir.ring.sort_index)

    def completeIncompleteRings(self):
        """This fcn takes each ir included in self and makes a closed path to
        use for its area computation."""

        # make sure ir_boolset is sorted (by sort index found in topological sort)
        self.sortIRs()

        if len(self.ir_boolset) == 0:
            return

        # iterate through IRs to complete them one by one, inner-most to outer-most
        for i,ir in enumerate(self.ir_boolset):
            # try all more-inner IRs (and the inner CR) starting with
            # most-outer among them - this is for finding transects.
            # Note: #poten[j] = ir_boolset[j-1].ring for j=1,...,i
            potential_rings = [self.inner] + [x.ring for x in self.ir_boolset[0:i]]

            # Find transects from the endpoints of ir to the next most-Outer
            # of the more-inner acceptable rings
            # Note: the ir's are sorted, but need to make sure each transect
            # is well-defined (i.e. check the potential connecting ir does in
            # fact travel "below" that endpoint)
            # Note: findTransects fcn used below will return
            # (False, False, False, False) for any transects already found
            nextRing2Try_index = i  # note this is right, not i-1 cause potential rings has self.inner at beginning
            while not (ir.transect0found and ir.transect1found) and nextRing2Try_index >= 0:
                nextRing2Try = potential_rings[nextRing2Try_index]
                if nextRing2Try_index == 0:
                    irORcr_2Try = self
                else:
                    irORcr_2Try = self.ir_boolset[nextRing2Try_index-1]

                tmp = ir.findTransects2endpointsFromInnerPath_normal(irORcr_2Try, nextRing2Try.path)
                (irORcr0, tL0, seg0, t0), (irORcr1, tL1, seg1, t1) = tmp

                # check if nextRing2Try is has a transect going to the
                # startpoint, if so we're done with this endpoint
                if tL0 != False:
                    # ring.path.point(T0) is where the transect, tL, meets
                    # this ring
                    T0 = segt2PathT(irORcr0.ORring.path,seg0,t0)

                    # record up-ladder on the ring the transect connects to
                    # (and t-val it connects at)
                    irORcr0.up_ladders.append((ir,T0))

                    # record startpoint down-ladder on this ir and (and t-val
                    # on connecting ring it connects at)
                    ir.down_ladder0 = (irORcr0,T0)
                    if not ir.transect0found:
                        ir.transect0found = True
                else:
                    ir.transect0fails.append(irORcr_2Try)

                # check if nextRing2Try has a transect going to the endpoint,
                # if so we're done with this endpoint
                if tL1 != False:
                    # ring.path.point(T0) is where the transect, tL, meets
                    # this ring
                    T1 = segt2PathT(irORcr1.ORring.path, seg1, t1)

                    # record up-ladder on the ring the transect connects to
                    # (and t-val it connects at)
                    irORcr1.up_ladders.append((ir, T1))

                    # record startpoint down_ladder on this ir and (and t-val
                    # on connecting ring it connects at)
                    ir.down_ladder1 = (irORcr1, T1)

                    if not ir.transect1found:
                        ir.transect1found = True
                else:
                    ir.transect1fails.append(irORcr_2Try)

                # unsuccessful while-loop termination conditions
                if (nextRing2Try_index == 0 and
                    not (ir.transect0found and ir.transect1found)):

                    printPath(ir.ring.path)
                    print(i)
                    colors = ['blue']*(len(self.ir_boolset)+2) + ['red']
                    paths2disp = ([self.inner.path] +
                                 [x.ring.path for x in self.ir_boolset] +
                                 [self.outer.path] +
                                 [ir.ring.path])
                    disvg(paths2disp, colors)
                    raise Exception("Acceptable more-inner ring could not be "
                                    "found.")
                else:
                    nextRing2Try_index -= 1

        # Now that all up/down ladders are set in this CR, iterate through IRs
        # again and create completed_path for each
        for ir in self.ir_boolset:
            try:
                ir.hardComplete()
            except:
                from options4rings import colordict
                # highlight ir in output SVG containing troubled section
                # (see area4rings)
                ir.ring.color = colordict['safe1']
                raise

#                #record this info to use after done with all IRs in this self
#                ir_info_list.append((ir,info0,info1))
#                ir_info_list.append((ir,ring0,irORcr0,seg0,t0,ring1,irORcr1,seg1,t1))
#            #now use this recorded info to complete the incomplete rings
#            for ir_item in ir_info_list:
#                (ir,info0,info1) = ir_item
#                (ring0,irORcr0,seg0,t0),(ring1,irORcr1,seg1,t1) = info0,info1
#                #create a new ring made of transect0 -> ir -> trans1->  partial ring from (closest path - make sure to transverse in correct direction)
#                if ring0 == ring1:
#    #                            try:
#                        ir.ezComplete_ir(ring0.path,seg0,t0,seg1,t1)
#    #                            except Exception as e:
#    #                                ir.completed_path = e
#                else:
#                    self.hardComplete_ir(ir,irORcr0,ring0.path,t0,T0,irORcr0,ring1.path,t1, T1)
#                            ir.completed_path = "There is an incomplete ring whose starting point is closest to a different ring than its ending point is closest to.  Handling this case is functionality Andy has yet to add, please let him know."
#                            raise Exception("There is an incomplete ring whose starting point is closest to a different ring than its ending point is closest to.  Handling this case is functionality Andy has yet to add, please let him know.")




    def minR(self):
        return self.inner.minR
    def maxR(self):
        return self.outer.maxR
    def aveR(self):
        return "Not Implemented"

    def add(self, value):
        self.ir_boolset.booladd(value)

#    def __repr__(self):
#        return '<CompleteRing containing %s incomplete rings: radius range = [%s,%s]>' %(len(self.ir_boolset),self.minR(),self.maxR())

    def __eq__(self, other):#just checks that inner ring is the same
        if not isinstance(other, CompleteRing):
            return NotImplemented
        if self.inner!=other.inner:
            return False
        return True

    def __ne__(self, other):
        if not isinstance(other, CompleteRing):
            return NotImplemented
        return not self == other

    def areaIgnoringIRs(self):
        return areaEnclosed(self.outer.path) - areaEnclosed(self.inner.path)

    def area(self):
        for iring in self.ir_boolset:
            if (not isinstance(iring.completed_path, Path)) or (iring.area() == "Fail"):
                return "Fail"
        return self.areaIgnoringIRs() - sum([iring.area() for iring in self.ir_boolset])

    def type(self, colordict):
        for (key,val) in colordict.items():
            if self.inner.color == val:
                innerkey = key
            if self.outer.color == val:
                outerkey = key
        try:
            return innerkey + "-" + outerkey
        except:
            raise Exception("inner ring color or outer ring color not in "
                            "colordict... you shouldn't have gotten this "
                            "far.  In other words, bug detected.")

#    def info(self, cp_index):
#        ###### "complete ring index, complete?, inner BrookID, outer BrookID, inner color, outer color, area, area Ignoring IRs, averageRadius, minR, maxR, IRs contained"
#        return str(cp_index) + "," + "True" + self.inner.brook_tag + ", " + self.outer.brook_tag + "," + self.inner.color + ", " + self.outer.color +"," + str(self.area()) +", "+ str(self.areaIgnoringIRs())+","+str(self.aveR())+","+str(self.minR)+","+str(self.maxR)+","+str(len(self.ir_boolset))

    def info(self,cp_index,colordict):
        ###### "complete ring index, type, # of IRs contained, minR, maxR, aveR, area, area Ignoring IRs"
        outputString = str(cp_index)+","+self.type(colordict)+","+str(len(self.ir_boolset))+","+str(self.minR())+","+str(self.maxR())+","+str(self.aveR())+","+str(self.area())+","+str(self.areaIgnoringIRs())
        for ir in self.ir_boolset:
            outputString += "\n"+ir.info(cp_index,colordict)
        return outputString


class CP_BoolSet(boolset):
    def cpUpdate(self,new_cp):
        for cp in self:
            if cp == new_cp:
                for ir in new_cp.ir_boolset:
                    cp.ir_boolset.booladd(ir)
                return
        self.append(new_cp)