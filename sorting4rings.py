from __future__ import division
from andysmod import format_time, topo_sorted, createOrderingMatrix, flattenList
from andysSVGpathTools import disvg, ptInsideClosedPath
from misc4rings import normalLineAtT_toInner_intersects_withOuter, pathXpathIntersections, display2rings4user, dis
from options4rings import basic_output_on, use_alternative_sorting_method, alt_sort_N, percentage_for_disagreement, look4ordering_matrices
import options4rings as opt
from svgpathtools import Path, CubicBezier,Line
from time import sleep, time as current_time
from itertools import combinations
from numpy import NaN, isnan, where, transpose
from operator import itemgetter
import cPickle as pickle
#def find_intersection_of_line_with_cubics_outward_pointing_normal_line_bundle(cub,line):
##The goal here is to find global max and min arguments of line (inside [0,1])
## at which there is a t in [0,1] s.t. the normal line of cub at t intersects
## with line
##Note: outward here is defined assuming path is oriented CCW
##OUTPUT: two tuples (t,s2) s.t. the normal line to cub at t intersecting with line at line.point(s2)
#    #set up coefficients for cub and line
#    a0,a1,a2,a3 = cub.start.real, cub.control1.real, cub.control2.real, cub.end.real
#    b0,b1,b2,b3 = cub.start.imag, cub.control1.imag, cub.control2.imag, cub.end.imag
#    q0r,q1r,q0i,q1i =line.start.real,line.end.real,line.start.imag,line.end.imag
#
#    #set up rational function to find zeros of
#    ndeg0 = (3*b0**3 + 9*b0*b1**2 - 3*b1**3 + (3*a0**2 - 8*a0*a1 + 3*a1**2 + 2*a0*a2)*b0 - (a0**2 - 6*a0*a1 + 3*a1**2 + 2*a0*a2 + 9*b0**2)*b1 - 2*(a0**2 - a0*a1)*b2)*q0i + (3*a0**3 - 9*a0**2*a1 + 9*a0*a1**2 - 3*a1**3 + (3*a0 - a1 - 2*a2)*b0**2 - 2*(4*a0 - 3*a1 - a2)*b0*b1 + 3*(a0 - a1)*b1**2 + 2*(a0 - a1)*b0*b2)*q0r - (3*b0**3 + 9*b0*b1**2 - 3*b1**3 + (3*a0**2 - 8*a0*a1 + 3*a1**2 + 2*a0*a2)*b0 - (a0**2 - 6*a0*a1 + 3*a1**2 + 2*a0*a2 + 9*b0**2)*b1 - 2*(a0**2 - a0*a1)*b2 + 2*((a1 - a2)*b0 - (a0 - a2)*b1 + (a0 - a1)*b2)*q0r)*q1i - (3*a0**3 - 9*a0**2*a1 + 9*a0*a1**2 - 3*a1**3 + (3*a0 - a1 - 2*a2)*b0**2 - 2*(4*a0 - 3*a1 - a2)*b0*b1 + 3*(a0 - a1)*b1**2 + 2*(a0 - a1)*b0*b2 - 2*((a1 - a2)*b0 - (a0 - a2)*b1 + (a0 - a1)*b2)*q0i)*q1r
#    ndeg1 = -2*(9*b0**3 + 45*b0*b1**2 - 18*b1**3 + (9*a0**2 - 29*a0*a1 + 18*a1**2 + 3*(4*a0 - 3*a1)*a2 - a0*a3)*b0 - (7*a0**2 - 27*a0*a1 + 18*a1**2 + 3*(4*a0 - 3*a1)*a2 - a0*a3 + 36*b0**2)*b1 - 3*(a0**2 - a0*a1 - 3*b0**2 + 6*b0*b1 - 3*b1**2)*b2 + (a0**2 - a0*a1)*b3)*q0i - 2*(9*a0**3 - 36*a0**2*a1 + 45*a0*a1**2 - 18*a1**3 + (9*a0 - 7*a1 - 3*a2 + a3)*b0**2 - (29*a0 - 27*a1 - 3*a2 + a3)*b0*b1 + 18*(a0 - a1)*b1**2 - (a0 - a1)*b0*b3 + 9*(a0**2 - 2*a0*a1 + a1**2)*a2 + 3*(4*(a0 - a1)*b0 - 3*(a0 - a1)*b1)*b2)*q0r + 2*(9*b0**3 + 45*b0*b1**2 - 18*b1**3 + (9*a0**2 - 29*a0*a1 + 18*a1**2 + 3*(4*a0 - 3*a1)*a2 - a0*a3)*b0 - (7*a0**2 - 27*a0*a1 + 18*a1**2 + 3*(4*a0 - 3*a1)*a2 - a0*a3 + 36*b0**2)*b1 - 3*(a0**2 - a0*a1 - 3*b0**2 + 6*b0*b1 - 3*b1**2)*b2 + (a0**2 - a0*a1)*b3 + ((2*a1 - 3*a2 + a3)*b0 - (2*a0 - 3*a2 + a3)*b1 + 3*(a0 - a1)*b2 - (a0 - a1)*b3)*q0r)*q1i + 2*(9*a0**3 - 36*a0**2*a1 + 45*a0*a1**2 - 18*a1**3 + (9*a0 - 7*a1 - 3*a2 + a3)*b0**2 - (29*a0 - 27*a1 - 3*a2 + a3)*b0*b1 + 18*(a0 - a1)*b1**2 - (a0 - a1)*b0*b3 + 9*(a0**2 - 2*a0*a1 + a1**2)*a2 + 3*(4*(a0 - a1)*b0 - 3*(a0 - a1)*b1)*b2 - ((2*a1 - 3*a2 + a3)*b0 - (2*a0 - 3*a2 + a3)*b1 + 3*(a0 - a1)*b2 - (a0 - a1)*b3)*q0i)*q1r
#    ndeg2 = (45*b0**3 + 351*b0*b1**2 - 171*b1**3 + 36*(b0 - b1)*b2**2 + (45*a0**2 - 170*a0*a1 + 141*a1**2 + 2*(47*a0 - 63*a1)*a2 + 18*a2**2 - 2*(7*a0 - 6*a1)*a3)*b0 - (55*a0**2 - 210*a0*a1 + 171*a1**2 + 6*(19*a0 - 24*a1)*a2 + 18*a2**2 - 4*(4*a0 - 3*a1)*a3 + 225*b0**2)*b1 + (5*a0**2 - 30*a0*a1 + 27*a1**2 + 18*(a0 - a1)*a2 - 2*a0*a3 + 99*b0**2 - 270*b0*b1 + 171*b1**2)*b2 + (5*a0**2 - 10*a0*a1 + 3*a1**2 + 2*a0*a2 - 9*b0**2 + 18*b0*b1 - 9*b1**2)*b3)*q0i + (45*a0**3 - 225*a0**2*a1 + 351*a0*a1**2 - 171*a1**3 + 36*(a0 - a1)*a2**2 + 5*(9*a0 - 11*a1 + a2 + a3)*b0**2 - 10*(17*a0 - 21*a1 + 3*a2 + a3)*b0*b1 + 3*(47*a0 - 57*a1 + 9*a2 + a3)*b1**2 + 18*(a0 - a1)*b2**2 + 9*(11*a0**2 - 30*a0*a1 + 19*a1**2)*a2 - 9*(a0**2 - 2*a0*a1 + a1**2)*a3 + 2*((47*a0 - 57*a1 + 9*a2 + a3)*b0 - 9*(7*a0 - 8*a1 + a2)*b1)*b2 - 2*((7*a0 - 8*a1 + a2)*b0 - 6*(a0 - a1)*b1)*b3)*q0r - (45*b0**3 + 351*b0*b1**2 - 171*b1**3 + 36*(b0 - b1)*b2**2 + (45*a0**2 - 170*a0*a1 + 141*a1**2 + 2*(47*a0 - 63*a1)*a2 + 18*a2**2 - 2*(7*a0 - 6*a1)*a3)*b0 - (55*a0**2 - 210*a0*a1 + 171*a1**2 + 6*(19*a0 - 24*a1)*a2 + 18*a2**2 - 4*(4*a0 - 3*a1)*a3 + 225*b0**2)*b1 + (5*a0**2 - 30*a0*a1 + 27*a1**2 + 18*(a0 - a1)*a2 - 2*a0*a3 + 99*b0**2 - 270*b0*b1 + 171*b1**2)*b2 + (5*a0**2 - 10*a0*a1 + 3*a1**2 + 2*a0*a2 - 9*b0**2 + 18*b0*b1 - 9*b1**2)*b3 + 2*((a1 - 2*a2 + a3)*b0 - (a0 - 3*a2 + 2*a3)*b1 + (2*a0 - 3*a1 + a3)*b2 - (a0 - 2*a1 + a2)*b3)*q0r)*q1i - (45*a0**3 - 225*a0**2*a1 + 351*a0*a1**2 - 171*a1**3 + 36*(a0 - a1)*a2**2 + 5*(9*a0 - 11*a1 + a2 + a3)*b0**2 - 10*(17*a0 - 21*a1 + 3*a2 + a3)*b0*b1 + 3*(47*a0 - 57*a1 + 9*a2 + a3)*b1**2 + 18*(a0 - a1)*b2**2 + 9*(11*a0**2 - 30*a0*a1 + 19*a1**2)*a2 - 9*(a0**2 - 2*a0*a1 + a1**2)*a3 + 2*((47*a0 - 57*a1 + 9*a2 + a3)*b0 - 9*(7*a0 - 8*a1 + a2)*b1)*b2 - 2*((7*a0 - 8*a1 + a2)*b0 - 6*(a0 - a1)*b1)*b3 - 2*((a1 - 2*a2 + a3)*b0 - (a0 - 3*a2 + 2*a3)*b1 + (2*a0 - 3*a1 + a3)*b2 - (a0 - 2*a1 + a2)*b3)*q0i)*q1r
#    ndeg3 = -4*(15*b0**3 + 171*b0*b1**2 - 102*b1**3 + 9*(5*b0 - 7*b1)*b2**2 + 6*b2**3 + (15*a0**2 - 65*a0*a1 + 66*a1**2 + (44*a0 - 81*a1)*a2 + 21*a2**2 - (9*a0 - 14*a1 + 5*a2)*a3)*b0 - (25*a0**2 - 105*a0*a1 + 102*a1**2 + (68*a0 - 117*a1)*a2 + 27*a2**2 - (13*a0 - 18*a1 + 5*a2)*a3 + 90*b0**2)*b1 + (10*a0**2 - 40*a0*a1 + 36*a1**2 + 12*(2*a0 - 3*a1)*a2 + 6*a2**2 - 4*(a0 - a1)*a3 + 54*b0**2 - 189*b0*b1 + 153*b1**2)*b2 - 9*(b0**2 - 3*b0*b1 + 2*b1**2 + (b0 - b1)*b2)*b3)*q0i - 4*(15*a0**3 - 90*a0**2*a1 + 171*a0*a1**2 - 102*a1**3 + 9*(5*a0 - 7*a1)*a2**2 + 6*a2**3 + 5*(3*a0 - 5*a1 + 2*a2)*b0**2 - 5*(13*a0 - 21*a1 + 8*a2)*b0*b1 + 6*(11*a0 - 17*a1 + 6*a2)*b1**2 + 3*(7*a0 - 9*a1 + 2*a2)*b2**2 + 9*(6*a0**2 - 21*a0*a1 + 17*a1**2)*a2 - 9*(a0**2 - 3*a0*a1 + 2*a1**2 + (a0 - a1)*a2)*a3 + (4*(11*a0 - 17*a1 + 6*a2)*b0 - 9*(9*a0 - 13*a1 + 4*a2)*b1)*b2 - ((9*a0 - 13*a1 + 4*a2)*b0 - 2*(7*a0 - 9*a1 + 2*a2)*b1 + 5*(a0 - a1)*b2)*b3)*q0r + 4*(15*b0**3 + 171*b0*b1**2 - 102*b1**3 + 9*(5*b0 - 7*b1)*b2**2 + 6*b2**3 + (15*a0**2 - 65*a0*a1 + 66*a1**2 + (44*a0 - 81*a1)*a2 + 21*a2**2 - (9*a0 - 14*a1 + 5*a2)*a3)*b0 - (25*a0**2 - 105*a0*a1 + 102*a1**2 + (68*a0 - 117*a1)*a2 + 27*a2**2 - (13*a0 - 18*a1 + 5*a2)*a3 + 90*b0**2)*b1 + (10*a0**2 - 40*a0*a1 + 36*a1**2 + 12*(2*a0 - 3*a1)*a2 + 6*a2**2 - 4*(a0 - a1)*a3 + 54*b0**2 - 189*b0*b1 + 153*b1**2)*b2 - 9*(b0**2 - 3*b0*b1 + 2*b1**2 + (b0 - b1)*b2)*b3)*q1i + 4*(15*a0**3 - 90*a0**2*a1 + 171*a0*a1**2 - 102*a1**3 + 9*(5*a0 - 7*a1)*a2**2 + 6*a2**3 + 5*(3*a0 - 5*a1 + 2*a2)*b0**2 - 5*(13*a0 - 21*a1 + 8*a2)*b0*b1 + 6*(11*a0 - 17*a1 + 6*a2)*b1**2 + 3*(7*a0 - 9*a1 + 2*a2)*b2**2 + 9*(6*a0**2 - 21*a0*a1 + 17*a1**2)*a2 - 9*(a0**2 - 3*a0*a1 + 2*a1**2 + (a0 - a1)*a2)*a3 + (4*(11*a0 - 17*a1 + 6*a2)*b0 - 9*(9*a0 - 13*a1 + 4*a2)*b1)*b2 - ((9*a0 - 13*a1 + 4*a2)*b0 - 2*(7*a0 - 9*a1 + 2*a2)*b1 + 5*(a0 - a1)*b2)*b3)*q1r
#    ndeg4 = (45*b0**3 + 711*b0*b1**2 - 513*b1**3 + 9*(37*b0 - 69*b1)*b2**2 + 108*b2**3 + 9*(b0 - b1)*b3**2 + (45*a0**2 - 220*a0*a1 + 261*a1**2 + 6*(29*a0 - 66*a1)*a2 + 141*a2**2 - 2*(22*a0 - 47*a1 + 30*a2)*a3 + 5*a3**2)*b0 - (95*a0**2 - 450*a0*a1 + 513*a1**2 + 18*(19*a0 - 41*a1)*a2 + 243*a2**2 - 2*(41*a0 - 81*a1 + 45*a2)*a3 + 5*a3**2 + 315*b0**2)*b1 + 6*(10*a0**2 - 45*a0*a1 + 48*a1**2 + (32*a0 - 63*a1)*a2 + 18*a2**2 - (7*a0 - 12*a1 + 5*a2)*a3 + 39*b0**2 - 168*b0*b1 + 171*b1**2)*b2 - 2*(5*a0**2 - 20*a0*a1 + 18*a1**2 + 6*(2*a0 - 3*a1)*a2 + 3*a2**2 - 2*(a0 - a1)*a3 + 27*b0**2 - 108*b0*b1 + 99*b1**2 + 9*(7*b0 - 11*b1)*b2 + 18*b2**2)*b3)*q0i + (45*a0**3 - 315*a0**2*a1 + 711*a0*a1**2 - 513*a1**3 + 9*(37*a0 - 69*a1)*a2**2 + 108*a2**3 + 9*(a0 - a1)*a3**2 + 5*(9*a0 - 19*a1 + 12*a2 - 2*a3)*b0**2 - 10*(22*a0 - 45*a1 + 27*a2 - 4*a3)*b0*b1 + 9*(29*a0 - 57*a1 + 32*a2 - 4*a3)*b1**2 + 3*(47*a0 - 81*a1 + 36*a2 - 2*a3)*b2**2 + 5*(a0 - a1)*b3**2 + 18*(13*a0**2 - 56*a0*a1 + 57*a1**2)*a2 - 18*(3*a0**2 - 12*a0*a1 + 11*a1**2 + (7*a0 - 11*a1)*a2 + 2*a2**2)*a3 + 6*((29*a0 - 57*a1 + 32*a2 - 4*a3)*b0 - 3*(22*a0 - 41*a1 + 21*a2 - 2*a3)*b1)*b2 - 2*((22*a0 - 41*a1 + 21*a2 - 2*a3)*b0 - (47*a0 - 81*a1 + 36*a2 - 2*a3)*b1 + 15*(2*a0 - 3*a1 + a2)*b2)*b3)*q0r - (45*b0**3 + 711*b0*b1**2 - 513*b1**3 + 9*(37*b0 - 69*b1)*b2**2 + 108*b2**3 + 9*(b0 - b1)*b3**2 + (45*a0**2 - 220*a0*a1 + 261*a1**2 + 6*(29*a0 - 66*a1)*a2 + 141*a2**2 - 2*(22*a0 - 47*a1 + 30*a2)*a3 + 5*a3**2)*b0 - (95*a0**2 - 450*a0*a1 + 513*a1**2 + 18*(19*a0 - 41*a1)*a2 + 243*a2**2 - 2*(41*a0 - 81*a1 + 45*a2)*a3 + 5*a3**2 + 315*b0**2)*b1 + 6*(10*a0**2 - 45*a0*a1 + 48*a1**2 + (32*a0 - 63*a1)*a2 + 18*a2**2 - (7*a0 - 12*a1 + 5*a2)*a3 + 39*b0**2 - 168*b0*b1 + 171*b1**2)*b2 - 2*(5*a0**2 - 20*a0*a1 + 18*a1**2 + 6*(2*a0 - 3*a1)*a2 + 3*a2**2 - 2*(a0 - a1)*a3 + 27*b0**2 - 108*b0*b1 + 99*b1**2 + 9*(7*b0 - 11*b1)*b2 + 18*b2**2)*b3)*q1i - (45*a0**3 - 315*a0**2*a1 + 711*a0*a1**2 - 513*a1**3 + 9*(37*a0 - 69*a1)*a2**2 + 108*a2**3 + 9*(a0 - a1)*a3**2 + 5*(9*a0 - 19*a1 + 12*a2 - 2*a3)*b0**2 - 10*(22*a0 - 45*a1 + 27*a2 - 4*a3)*b0*b1 + 9*(29*a0 - 57*a1 + 32*a2 - 4*a3)*b1**2 + 3*(47*a0 - 81*a1 + 36*a2 - 2*a3)*b2**2 + 5*(a0 - a1)*b3**2 + 18*(13*a0**2 - 56*a0*a1 + 57*a1**2)*a2 - 18*(3*a0**2 - 12*a0*a1 + 11*a1**2 + (7*a0 - 11*a1)*a2 + 2*a2**2)*a3 + 6*((29*a0 - 57*a1 + 32*a2 - 4*a3)*b0 - 3*(22*a0 - 41*a1 + 21*a2 - 2*a3)*b1)*b2 - 2*((22*a0 - 41*a1 + 21*a2 - 2*a3)*b0 - (47*a0 - 81*a1 + 36*a2 - 2*a3)*b1 + 15*(2*a0 - 3*a1 + a2)*b2)*b3)*q1r
#    ndeg5 = -2*(9*b0**3 + 189*b0*b1**2 - 162*b1**3 + 27*(5*b0 - 12*b1)*b2**2 + 81*b2**3 + 9*(b0 - 2*b1 + b2)*b3**2 + (9*a0**2 - 49*a0*a1 + 66*a1**2 + (44*a0 - 117*a1)*a2 + 51*a2**2 - (13*a0 - 34*a1 + 29*a2)*a3 + 4*a3**2)*b0 - (23*a0**2 - 123*a0*a1 + 162*a1**2 + 9*(12*a0 - 31*a1)*a2 + 117*a2**2 - (31*a0 - 78*a1 + 63*a2)*a3 + 8*a3**2 + 72*b0**2)*b1 + (19*a0**2 - 99*a0*a1 + 126*a1**2 + 3*(28*a0 - 69*a1)*a2 + 81*a2**2 - (23*a0 - 54*a1 + 39*a2)*a3 + 4*a3**2 + 63*b0**2 - 324*b0*b1 + 405*b1**2)*b2 - (5*a0**2 - 25*a0*a1 + 30*a1**2 + 5*(4*a0 - 9*a1)*a2 + 15*a2**2 - 5*(a0 - 2*a1 + a2)*a3 + 18*b0**2 - 90*b0*b1 + 108*b1**2 + 18*(4*b0 - 9*b1)*b2 + 54*b2**2)*b3)*q0i - 2*(9*a0**3 - 72*a0**2*a1 + 189*a0*a1**2 - 162*a1**3 + 27*(5*a0 - 12*a1)*a2**2 + 81*a2**3 + 9*(a0 - 2*a1 + a2)*a3**2 + (9*a0 - 23*a1 + 19*a2 - 5*a3)*b0**2 - (49*a0 - 123*a1 + 99*a2 - 25*a3)*b0*b1 + 6*(11*a0 - 27*a1 + 21*a2 - 5*a3)*b1**2 + 3*(17*a0 - 39*a1 + 27*a2 - 5*a3)*b2**2 + 4*(a0 - 2*a1 + a2)*b3**2 + 9*(7*a0**2 - 36*a0*a1 + 45*a1**2)*a2 - 18*(a0**2 - 5*a0*a1 + 6*a1**2 + (4*a0 - 9*a1)*a2 + 3*a2**2)*a3 + (4*(11*a0 - 27*a1 + 21*a2 - 5*a3)*b0 - 9*(13*a0 - 31*a1 + 23*a2 - 5*a3)*b1)*b2 - ((13*a0 - 31*a1 + 23*a2 - 5*a3)*b0 - 2*(17*a0 - 39*a1 + 27*a2 - 5*a3)*b1 + (29*a0 - 63*a1 + 39*a2 - 5*a3)*b2)*b3)*q0r + 2*(9*b0**3 + 189*b0*b1**2 - 162*b1**3 + 27*(5*b0 - 12*b1)*b2**2 + 81*b2**3 + 9*(b0 - 2*b1 + b2)*b3**2 + (9*a0**2 - 49*a0*a1 + 66*a1**2 + (44*a0 - 117*a1)*a2 + 51*a2**2 - (13*a0 - 34*a1 + 29*a2)*a3 + 4*a3**2)*b0 - (23*a0**2 - 123*a0*a1 + 162*a1**2 + 9*(12*a0 - 31*a1)*a2 + 117*a2**2 - (31*a0 - 78*a1 + 63*a2)*a3 + 8*a3**2 + 72*b0**2)*b1 + (19*a0**2 - 99*a0*a1 + 126*a1**2 + 3*(28*a0 - 69*a1)*a2 + 81*a2**2 - (23*a0 - 54*a1 + 39*a2)*a3 + 4*a3**2 + 63*b0**2 - 324*b0*b1 + 405*b1**2)*b2 - (5*a0**2 - 25*a0*a1 + 30*a1**2 + 5*(4*a0 - 9*a1)*a2 + 15*a2**2 - 5*(a0 - 2*a1 + a2)*a3 + 18*b0**2 - 90*b0*b1 + 108*b1**2 + 18*(4*b0 - 9*b1)*b2 + 54*b2**2)*b3)*q1i + 2*(9*a0**3 - 72*a0**2*a1 + 189*a0*a1**2 - 162*a1**3 + 27*(5*a0 - 12*a1)*a2**2 + 81*a2**3 + 9*(a0 - 2*a1 + a2)*a3**2 + (9*a0 - 23*a1 + 19*a2 - 5*a3)*b0**2 - (49*a0 - 123*a1 + 99*a2 - 25*a3)*b0*b1 + 6*(11*a0 - 27*a1 + 21*a2 - 5*a3)*b1**2 + 3*(17*a0 - 39*a1 + 27*a2 - 5*a3)*b2**2 + 4*(a0 - 2*a1 + a2)*b3**2 + 9*(7*a0**2 - 36*a0*a1 + 45*a1**2)*a2 - 18*(a0**2 - 5*a0*a1 + 6*a1**2 + (4*a0 - 9*a1)*a2 + 3*a2**2)*a3 + (4*(11*a0 - 27*a1 + 21*a2 - 5*a3)*b0 - 9*(13*a0 - 31*a1 + 23*a2 - 5*a3)*b1)*b2 - ((13*a0 - 31*a1 + 23*a2 - 5*a3)*b0 - 2*(17*a0 - 39*a1 + 27*a2 - 5*a3)*b1 + (29*a0 - 63*a1 + 39*a2 - 5*a3)*b2)*b3)*q1r
#    ndeg6 = 3*(b0**3 + 27*b0*b1**2 - 27*b1**3 + 27*(b0 - 3*b1)*b2**2 + 27*b2**3 + 3*(b0 - 3*b1 + 3*b2)*b3**2 - b3**3 + (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2)*b0 - 3*(a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2 + 3*b0**2)*b1 + 3*(a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2 + 3*b0**2 - 18*b0*b1 + 27*b1**2)*b2 - (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2 + 3*b0**2 - 18*b0*b1 + 27*b1**2 + 18*(b0 - 3*b1)*b2 + 27*b2**2)*b3)*q0i + 3*(a0**3 - 9*a0**2*a1 + 27*a0*a1**2 - 27*a1**3 + 27*(a0 - 3*a1)*a2**2 + 27*a2**3 + 3*(a0 - 3*a1 + 3*a2)*a3**2 - a3**3 + (a0 - 3*a1 + 3*a2 - a3)*b0**2 - 6*(a0 - 3*a1 + 3*a2 - a3)*b0*b1 + 9*(a0 - 3*a1 + 3*a2 - a3)*b1**2 + 9*(a0 - 3*a1 + 3*a2 - a3)*b2**2 + (a0 - 3*a1 + 3*a2 - a3)*b3**2 + 9*(a0**2 - 6*a0*a1 + 9*a1**2)*a2 - 3*(a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2)*a3 + 6*((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1)*b2 - 2*((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1 + 3*(a0 - 3*a1 + 3*a2 - a3)*b2)*b3)*q0r - 3*(b0**3 + 27*b0*b1**2 - 27*b1**3 + 27*(b0 - 3*b1)*b2**2 + 27*b2**3 + 3*(b0 - 3*b1 + 3*b2)*b3**2 - b3**3 + (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2)*b0 - 3*(a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2 + 3*b0**2)*b1 + 3*(a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2 + 3*b0**2 - 18*b0*b1 + 27*b1**2)*b2 - (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2 + 3*b0**2 - 18*b0*b1 + 27*b1**2 + 18*(b0 - 3*b1)*b2 + 27*b2**2)*b3)*q1i - 3*(a0**3 - 9*a0**2*a1 + 27*a0*a1**2 - 27*a1**3 + 27*(a0 - 3*a1)*a2**2 + 27*a2**3 + 3*(a0 - 3*a1 + 3*a2)*a3**2 - a3**3 + (a0 - 3*a1 + 3*a2 - a3)*b0**2 - 6*(a0 - 3*a1 + 3*a2 - a3)*b0*b1 + 9*(a0 - 3*a1 + 3*a2 - a3)*b1**2 + 9*(a0 - 3*a1 + 3*a2 - a3)*b2**2 + (a0 - 3*a1 + 3*a2 - a3)*b3**2 + 9*(a0**2 - 6*a0*a1 + 9*a1**2)*a2 - 3*(a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2)*a3 + 6*((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1)*b2 - 2*((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1 + 3*(a0 - 3*a1 + 3*a2 - a3)*b2)*b3)*q1r
#    numer_coeffs = (ndeg6,ndeg5,ndeg4,ndeg3,ndeg2,ndeg1,ndeg0)
#    ddeg0 = (b0**2 - 2*b0*b1 + b1**2)*q0i**2 + 2*((a0 - a1)*b0 - (a0 - a1)*b1)*q0i*q0r + (a0**2 - 2*a0*a1 + a1**2)*q0r**2 + (b0**2 - 2*b0*b1 + b1**2)*q1i**2 + (a0**2 - 2*a0*a1 + a1**2)*q1r**2 - 2*((b0**2 - 2*b0*b1 + b1**2)*q0i + ((a0 - a1)*b0 - (a0 - a1)*b1)*q0r)*q1i - 2*(((a0 - a1)*b0 - (a0 - a1)*b1)*q0i + (a0**2 - 2*a0*a1 + a1**2)*q0r - ((a0 - a1)*b0 - (a0 - a1)*b1)*q1i)*q1r
#    ddeg1 = -4*(b0**2 - 3*b0*b1 + 2*b1**2 + (b0 - b1)*b2)*q0i**2 - 4*((2*a0 - 3*a1 + a2)*b0 - (3*a0 - 4*a1 + a2)*b1 + (a0 - a1)*b2)*q0i*q0r - 4*(a0**2 - 3*a0*a1 + 2*a1**2 + (a0 - a1)*a2)*q0r**2 - 4*(b0**2 - 3*b0*b1 + 2*b1**2 + (b0 - b1)*b2)*q1i**2 - 4*(a0**2 - 3*a0*a1 + 2*a1**2 + (a0 - a1)*a2)*q1r**2 + 4*(2*(b0**2 - 3*b0*b1 + 2*b1**2 + (b0 - b1)*b2)*q0i + ((2*a0 - 3*a1 + a2)*b0 - (3*a0 - 4*a1 + a2)*b1 + (a0 - a1)*b2)*q0r)*q1i + 4*(((2*a0 - 3*a1 + a2)*b0 - (3*a0 - 4*a1 + a2)*b1 + (a0 - a1)*b2)*q0i + 2*(a0**2 - 3*a0*a1 + 2*a1**2 + (a0 - a1)*a2)*q0r - ((2*a0 - 3*a1 + a2)*b0 - (3*a0 - 4*a1 + a2)*b1 + (a0 - a1)*b2)*q1i)*q1r
#    ddeg2 = 2*(3*b0**2 - 12*b0*b1 + 11*b1**2 + (7*b0 - 11*b1)*b2 + 2*b2**2 - (b0 - b1)*b3)*q0i**2 + 2*((6*a0 - 12*a1 + 7*a2 - a3)*b0 - (12*a0 - 22*a1 + 11*a2 - a3)*b1 + (7*a0 - 11*a1 + 4*a2)*b2 - (a0 - a1)*b3)*q0i*q0r + 2*(3*a0**2 - 12*a0*a1 + 11*a1**2 + (7*a0 - 11*a1)*a2 + 2*a2**2 - (a0 - a1)*a3)*q0r**2 + 2*(3*b0**2 - 12*b0*b1 + 11*b1**2 + (7*b0 - 11*b1)*b2 + 2*b2**2 - (b0 - b1)*b3)*q1i**2 + 2*(3*a0**2 - 12*a0*a1 + 11*a1**2 + (7*a0 - 11*a1)*a2 + 2*a2**2 - (a0 - a1)*a3)*q1r**2 - 2*(2*(3*b0**2 - 12*b0*b1 + 11*b1**2 + (7*b0 - 11*b1)*b2 + 2*b2**2 - (b0 - b1)*b3)*q0i + ((6*a0 - 12*a1 + 7*a2 - a3)*b0 - (12*a0 - 22*a1 + 11*a2 - a3)*b1 + (7*a0 - 11*a1 + 4*a2)*b2 - (a0 - a1)*b3)*q0r)*q1i - 2*(((6*a0 - 12*a1 + 7*a2 - a3)*b0 - (12*a0 - 22*a1 + 11*a2 - a3)*b1 + (7*a0 - 11*a1 + 4*a2)*b2 - (a0 - a1)*b3)*q0i + 2*(3*a0**2 - 12*a0*a1 + 11*a1**2 + (7*a0 - 11*a1)*a2 + 2*a2**2 - (a0 - a1)*a3)*q0r - ((6*a0 - 12*a1 + 7*a2 - a3)*b0 - (12*a0 - 22*a1 + 11*a2 - a3)*b1 + (7*a0 - 11*a1 + 4*a2)*b2 - (a0 - a1)*b3)*q1i)*q1r
#    ddeg3 = -4*(b0**2 - 5*b0*b1 + 6*b1**2 + (4*b0 - 9*b1)*b2 + 3*b2**2 - (b0 - 2*b1 + b2)*b3)*q0i**2 - 4*((2*a0 - 5*a1 + 4*a2 - a3)*b0 - (5*a0 - 12*a1 + 9*a2 - 2*a3)*b1 + (4*a0 - 9*a1 + 6*a2 - a3)*b2 - (a0 - 2*a1 + a2)*b3)*q0i*q0r - 4*(a0**2 - 5*a0*a1 + 6*a1**2 + (4*a0 - 9*a1)*a2 + 3*a2**2 - (a0 - 2*a1 + a2)*a3)*q0r**2 - 4*(b0**2 - 5*b0*b1 + 6*b1**2 + (4*b0 - 9*b1)*b2 + 3*b2**2 - (b0 - 2*b1 + b2)*b3)*q1i**2 - 4*(a0**2 - 5*a0*a1 + 6*a1**2 + (4*a0 - 9*a1)*a2 + 3*a2**2 - (a0 - 2*a1 + a2)*a3)*q1r**2 + 4*(2*(b0**2 - 5*b0*b1 + 6*b1**2 + (4*b0 - 9*b1)*b2 + 3*b2**2 - (b0 - 2*b1 + b2)*b3)*q0i + ((2*a0 - 5*a1 + 4*a2 - a3)*b0 - (5*a0 - 12*a1 + 9*a2 - 2*a3)*b1 + (4*a0 - 9*a1 + 6*a2 - a3)*b2 - (a0 - 2*a1 + a2)*b3)*q0r)*q1i + 4*(((2*a0 - 5*a1 + 4*a2 - a3)*b0 - (5*a0 - 12*a1 + 9*a2 - 2*a3)*b1 + (4*a0 - 9*a1 + 6*a2 - a3)*b2 - (a0 - 2*a1 + a2)*b3)*q0i + 2*(a0**2 - 5*a0*a1 + 6*a1**2 + (4*a0 - 9*a1)*a2 + 3*a2**2 - (a0 - 2*a1 + a2)*a3)*q0r - ((2*a0 - 5*a1 + 4*a2 - a3)*b0 - (5*a0 - 12*a1 + 9*a2 - 2*a3)*b1 + (4*a0 - 9*a1 + 6*a2 - a3)*b2 - (a0 - 2*a1 + a2)*b3)*q1i)*q1r
#    ddeg4 = (b0**2 - 6*b0*b1 + 9*b1**2 + 6*(b0 - 3*b1)*b2 + 9*b2**2 - 2*(b0 - 3*b1 + 3*b2)*b3 + b3**2)*q0i**2 + 2*((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1 + 3*(a0 - 3*a1 + 3*a2 - a3)*b2 - (a0 - 3*a1 + 3*a2 - a3)*b3)*q0i*q0r + (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2)*q0r**2 + (b0**2 - 6*b0*b1 + 9*b1**2 + 6*(b0 - 3*b1)*b2 + 9*b2**2 - 2*(b0 - 3*b1 + 3*b2)*b3 + b3**2)*q1i**2 + (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2)*q1r**2 - 2*((b0**2 - 6*b0*b1 + 9*b1**2 + 6*(b0 - 3*b1)*b2 + 9*b2**2 - 2*(b0 - 3*b1 + 3*b2)*b3 + b3**2)*q0i + ((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1 + 3*(a0 - 3*a1 + 3*a2 - a3)*b2 - (a0 - 3*a1 + 3*a2 - a3)*b3)*q0r)*q1i - 2*(((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1 + 3*(a0 - 3*a1 + 3*a2 - a3)*b2 - (a0 - 3*a1 + 3*a2 - a3)*b3)*q0i + (a0**2 - 6*a0*a1 + 9*a1**2 + 6*(a0 - 3*a1)*a2 + 9*a2**2 - 2*(a0 - 3*a1 + 3*a2)*a3 + a3**2)*q0r - ((a0 - 3*a1 + 3*a2 - a3)*b0 - 3*(a0 - 3*a1 + 3*a2 - a3)*b1 + 3*(a0 - 3*a1 + 3*a2 - a3)*b2 - (a0 - 3*a1 + 3*a2 - a3)*b3)*q1i)*q1r
#    denom_coeffs = (ddeg4,ddeg3,ddeg2,ddeg1,ddeg0)
#
#    #normal line distance function
#    s2 = lambda t: (b0*(t - 1)**3 - 3*b1*(t - 1)**2*t + 3*b2*(t - 1)*t**2 - b3*t**3 + (a0*(t - 1)**3 - 3*a1*(t - 1)**2*t + 3*a2*(t - 1)*t**2 - a3*t**3 + q0r)*(a0*(t - 1)**2 - a1*(t - 1)**2 - 2*a1*(t - 1)*t + 2*a2*(t - 1)*t + a2*t**2 - a3*t**2)/(b0*(t - 1)**2 - b1*(t - 1)**2 - 2*b1*(t - 1)*t + 2*b2*(t - 1)*t + b2*t**2 - b3*t**2) + q0i)/(q0i + (a0*(t - 1)**2 - a1*(t - 1)**2 - 2*a1*(t - 1)*t + 2*a2*(t - 1)*t + a2*t**2 - a3*t**2)*(q0r - q1r)/(b0*(t - 1)**2 - b1*(t - 1)**2 - 2*b1*(t - 1)*t + 2*b2*(t - 1)*t + b2*t**2 - b3*t**2) - q1i)
#
#    #Find zeros
#    numer_roots = roots(numer_coeffs)
#    denom_roots = roots(denom_coeffs)
#
#    #get rid of duplicates, non-real, and out-of-bounds t-vals
#    extremizers = set([t.real for t in numer_roots if 0<=t<=1 and isclose(t.imag,0)])
#    undef_pts = set([t.real for t in denom_roots if 0<=t<=1 and isclose(t.imag,0)])
#
#    #get rid of t-vals where rational fcn is undefined  ###TOL using numpy.isclose()
#    for t_denom in undef_pts:
#        for t_numer in extremizers:
#            if isclose(t_numer,t_denom):
#                extremizers.remove(t_numer)
#
#    #Make list of extrema corresponding to extremizers  ###TOL using numpy.isclose()
#    bdry_extrema = []
#    if not isclose(0,-3*(b0 - b1)*(q0i - q1i) - 3*(a0 - a1)*(q0r - q1r)):
#        start_norm_dist = -((a0 - a1)*(a0 - q0r)/(b0 - b1) + b0 - q0i)/(q0i + (a0 - a1)*(q0r - q1r)/(b0 - b1) - q1i)
#        bdry_extrema.append((0,start_norm_dist))
#    if not isclose(0,-3*(b2 - b3)*(q0i - q1i) - 3*(a2 - a3)*(q0r - q1r)):
#        end_norm_dist = -((a2 - a3)*(a3 - q0r)/(b2 - b3) + b3 - q0i)/(q0i + (a2 - a3)*(q0r - q1r)/(b2 - b3) - q1i)
#        bdry_extrema.append((1,end_norm_dist))
#    extrema = [(t,s2(t)) for t in extremizers]+bdry_extrema
#    extrema = [(t,s2_val) for (t,s2_val) in extrema if 0<=s2_val<=1]
#
#    #get rid of pairs where normal points inward
#    extrema_copy = copy(extrema)
#    for (t,s2_val) in extrema_copy:
#        nL_vec1 = line.point(s2_val)-cub.point(t)
#        nL_vec2 = -1j*segDerivative(cub,t)
#        dot_prod = (nL_vec1*nL_vec2.conjugate()).real
#        if dot_prod < 0:
#            extrema.remove((t,s2_val))
#
#
#    #Find global extrema over interval [0,1]
#    if extrema:
#        maximum = max(extrema,key=itemgetter(1))
#        minimum = min(extrema,key=itemgetter(1))
#    else:
#        maximum = False
#        minimum = False
#    return minimum, maximum
#def find_intersection_of_rectangle_with_paths_outward_pointing_normal_line_bundle(path,rectangular_path):
##Note: outward here is defined assuming path is oriented CCW
#    extremizers = []
#    for line in rectangular_path:
#        for seg in path:
#            if isinstance(seg,CubicBezier):
#                seg_as_cub = seg
#                mini,maxi = find_intersection_of_line_with_cubics_outward_pointing_normal_line_bundle(seg,line)
#            elif isinstance(seg,Line): ###This could likely be improved speed-wise by not converting seg to cub and doing the math out for a line
#                seg_as_cub = CubicBezier(seg.start,seg.start,seg.end,seg.end)
#                mini,maxi = find_intersection_of_line_with_cubics_outward_pointing_normal_line_bundle(seg_as_cub,line)
#            else:
#                raise Exception("A segment from path is neither a Line nor a CubicBezier object.")
#            if mini:
#                minimum = (segt2PathT(path,seg_as_cub,mini[0]),segt2PathT(rectangular_path,line,mini[1]))
#                maximum = (segt2PathT(path,seg_as_cub,maxi[0]),segt2PathT(rectangular_path,line,maxi[1]))
#                extremizers += [minimum,maximum]
#    distfcn =lambda x: x[0][0] - x[1][0]
#    ((T_0,S2_0),(T_1,S2_1)) = max((((T_a,S2_a),(T_b,S2_b)) for ((T_a,S2_a),(T_b,S2_b)) in combinations(extremizers,2)),key=distfcn)
#    return ((T_0,S2_0),(T_1,S2_1))


def ring1_isbelow_ring2_numHits(ring1,ring2,N,debug_name=''): #replaced by normal transect based method
    center = ring1.center
    countHits = 0 # number (out of N) of the checked lines from ir1 to the center that intersected  with ir2
    tran_colors = [] #for debug mode
    tran_lines = [] #for debug mode
    for i in range(N):
        innerT = i/(N-1)
        nlin, seg_out, t_out = normalLineAtT_toInner_intersects_withOuter(innerT,ring1.path,ring2.path,center,'debug')
        if debug_name != '': #output an SVG with the lines used
            tran_lines.append(nlin)
            if seg_out !=False:
                tran_colors.append('black')
            else:
                tran_colors.append('purple')
        if seg_out != False:
            countHits += 1
    if debug_name != '':
        dis([ring1.path,ring2.path],['green','red'],lines=tran_lines,line_colors=tran_colors,filename=debug_name)
    return countHits


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


def ask_user_to_sort(i,j,ring_list,make_svg=True,ask_later=True):#returns 1 if ring_list[i] is the more inner ring, -1 if ring_list[j] is and 0 if they are incomparable (or equal)
#    if i>j:
#        return -1*ask_user_to_sort(j,i,ring_list) #prevents asking user about same set of rings twice
    if ask_later: #save an svg for "interactive sorting" and return NaN
        from options4rings import outputFolder
        from os import path as os_path
        save_loc = os_path.join(outputFolder,'interactive_sorting',
                                ring_list[i].svgname,'cmp_%s-%s.svg'%(i,j))
        disvg([ring_list[i].path,ring_list[j].path],['green','red'],
              nodes=[ring_list[i].center],filename=save_loc)
        return NaN
    else:
        if make_svg:
            display2rings4user(i,j,ring_list)
        try: input = raw_input #this try/except is for python 3 compatibility
        except NameError: pass
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
                                  N_lines2use=alt_sort_N, increase_N_if_zero=True):#####TOL
    """Returns 1 if true, -1 if false and 0 if equal"""
    ring1 = ringlist[ring1_index]
    ring2 = ringlist[ring2_index]
    if ring1.path==ring2.path:
        return 0
    countHits12 = ring1_isbelow_ring2_numHits(ring1, ring2, N_lines2use)
    countHits21 = ring1_isbelow_ring2_numHits(ring2, ring1, N_lines2use)
    if countHits12==0 or countHits21==0:
        if countHits12>0:
            return -1
        elif countHits21>0:
            return 1
        elif increase_N_if_zero:
            N_upped = N_lines2use * max(len(ring1.path), len(ring2.path))
            improved_res = ring1_isoutside_ring2_cmp_alt(ringlist, ring1_index, ring2_index, N_lines2use=N_upped, increase_N_if_zero=False)
            if improved_res != 0:
                return improved_res
            elif ring1.isClosed() or ring2.isClosed():
                if opt.manually_fix_sorting:
                    return ask_user_to_sort(ring1_index, ring2_index, ringlist, make_svg=True)
                else:
                    raise Exception("Problem sorting rings... set 'manually_fix_sorting=True' in options4rings.py to fix manually.")
            else:
                return 0
        else:
            return 0

    #neither of the counts were zero
    ratio21over12 = countHits21/countHits12
    try:
        upper_bound = 1.0/percentage_for_disagreement
    except ZeroDivisionError:
        from numpy import Inf
        upper_bound = Inf

    if percentage_for_disagreement <ratio21over12< upper_bound:  # not sure... use more lines
        N_upped = N_lines2use * max(len(ring1.path), len(ring2.path))
        countHits12 = ring1_isbelow_ring2_numHits(ring1, ring2, N_upped)
        countHits21 = ring1_isbelow_ring2_numHits(ring2, ring1, N_upped)
        ratio21over12 = countHits21/countHits12
        if percentage_for_disagreement < ratio21over12 < upper_bound:  # still not sure, ask user, if allowed
            if opt.manually_fix_sorting:
                return ask_user_to_sort(ring1_index, ring2_index, ringlist, make_svg=True)
            else:
                raise Exception("Problem sorting rings... set 'manually_fix_sorting=True' in options4rings.py to fix manually.")
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
        r1_must_be_outside_r2 = ptInsideClosedPath(r1_pt,outside_point,ring2.path_around_bdry(bdry_path))
        r2_must_be_outside_r1 = ptInsideClosedPath(r2_pt,outside_point,ring1.path_around_bdry(bdry_path))
        if r1_must_be_outside_r2 and not r2_must_be_outside_r1:
            return 1
        elif not r1_must_be_outside_r2 and r2_must_be_outside_r1:
            return -1
        elif not r1_must_be_outside_r2 and not r2_must_be_outside_r1:
            return 0
        else:
            raise Exception("This case should never be reached.")


class Closed_Pair(object):

    def __init__(self, ring_list, outside_point, inner, outer,  contents=None):
        if not contents:
            contents = []
        self.outside_point  = outside_point#a point known to be outside this cp
        self.inner_index = inner
        self.outer_index = outer
        self.contents = contents #list of indices of open ring contained between inner and outer rings
        self.contents_psorting = None
        self.ring_list = ring_list
        self.isCore = (inner=='core')

    def contains(self,or_index): #note all intersection between rings have already been removed
        oring = self.ring_list[or_index]
        if len(oring.path)>2:
            pt = oring.path[1].start
        else:
            pt = oring.path.point(0.5)
        outer = self.ring_list[self.outer_index]
        if self.isCore: #if the inner ring of the Closed_Pair is actually the center
            if oring.maxR > outer.maxR:
                return False
            else:
                if ptInsideClosedPath(pt,self.outside_point,outer.path):
                    return True
                else:
                    return False
        else:
            inner = self.ring_list[self.inner_index]
            if oring.maxR> outer.maxR or oring.minR < inner.minR:
                return False
            else:
                if ptInsideClosedPath(pt,self.outside_point,outer.path) and not ptInsideClosedPath(pt,self.outside_point,inner.path):
                    return True
                else:
                    return False

    def __str__(self):
        return "{(%s,%s): %s}"%(self.inner_index,self.outer_index,self.contents)

    def __repr__(self):
        return str(self)


def sort_rings(ring_list, om_pickle_file):
    #make list of pairs of consecutive closed rings
    basic_output_on.dprint("\nSorting closed rings...",'nr')
    bdry_ring = max(ring_list,key=lambda rg: rg.maxR)
    outside_point = bdry_ring.center + 2*bdry_ring.maxR #a point known to be outside all rings
    maxRkey = lambda idx: ring_list[idx].maxR
    sorted_closedRingIndices = ['core'] + sorted([rl_ind for rl_ind,r in enumerate(ring_list) if r.isClosed()],key=maxRkey)
    closed_pairs = [Closed_Pair(ring_list, outside_point, sorted_closedRingIndices[k-1], sorted_closedRingIndices[k]) for k in xrange(1,len(sorted_closedRingIndices))]

#    ###DEBUG ONLY TEST slideshow (of closed rings)
#    print [c.inner_index for c in closed_pairs]
#    print [c.outer_index for c in closed_pairs]
#    n = len(sorted_closedRingIndices) ###DEBUG
#    disvg([ring_list[c.outer_index].path for c in closed_pairs if c.outer_index!='core'],[ring_list[c.outer_index].color for c in closed_pairs if c.outer_index!='core'],openInBrowser=True)###DEBUG
#
#    disvg([ring_list[k].path for k in sorted_closedRingIndices[1:n]],[ring_list[k].color for k in sorted_closedRingIndices[1:n]],openInBrowser=True)###DEBUG
#    from os import path as os_path
#    from options4rings import outputFolder
#    from andysSVGpathTools import svgSlideShow
#    save_dir = os_path.join(outputFolder,'debug','ring_sort_slideshow_closed_pairs')
#
#    used_rings = []
#    pathcolortuplelist = []
#    paths = [ring_list[k].path for k in sorted_closedRingIndices if k!='core']
#    for k,cp in enumerate(closed_pairs):
#        colors = []
#        for i in sorted_closedRingIndices:
#            if i in used_rings:
#                colors.append('yellow')
#            elif i==cp.inner_index:
#                colors.append('green')
#            elif i==cp.outer_index:
#                colors.append('red')
#            else:
#                colors.append('blue')
#        print "%s : %s : %s"%(k,cp,colors)
#        pathcolortuplelist.append((paths,colors))
#        used_rings = used_rings+[cp.inner_index]
#    print
#    svgSlideShow(pathcolortuplelist,save_directory=save_dir,clear_directory=True,suppressOutput=False)
#    ###End of DEBUG ONLY TEST slideshow (of closed rings)

    #Find the lines to the boundary and the path given
    if not use_alternative_sorting_method:
        center = ring_list[0].center
        d = 1.5*bdry_ring.maxR
        pts = [center-d+d*1j, center-d-d*1j, center+d-d*1j, center+d+d*1j]
        rectangle_containing_bdry = Path(*[Line(pts[i],pts[(i+1)%4]) for i in range(4)])
        for r in ring_list:
            if not r.isClosed():
                r.findLines2Bdry(rectangle_containing_bdry)

    #figure out which open (incomplete) rings live between which closed rings
    basic_output_on.dprint("Done, closed rings sorted.\nNow determining which open rings lie between which closed pairs of rings...",'nr')
    start_time = current_time()
    cur_time = start_time
    unlocated_openRingIndices = [i for i,r in enumerate(ring_list) if not r.isClosed()]
#    N_open = len(unlocated_openRingIndices)
    for cp_idx,cp in enumerate(closed_pairs):
        #Progess
        N_unlocated = len(unlocated_openRingIndices)
#        basic_output_on.dprint("\rWorking on Closed_Pair (cp) %s / %s ... %s / %s open rings placed.  Last cp: %s | Total Elapsed: %s"%(cp_idx+1,len(closed_pairs),N_open-N_unlocated,N_open,format_time(current_time()-cur_time),format_time(current_time()-start_time)),'nr')
        cur_time = current_time()

        #Loop through unlocated open rings
        located_in_this_cp = []
        for idx_un in range(N_unlocated):
            or_index = unlocated_openRingIndices[idx_un]
            cp_contains_or = cp.contains(or_index)

            if cp_contains_or:
                cp.contents.append(or_index)
                located_in_this_cp.append(idx_un)
        #delete those unlocated open rings that were located in this cp
        unlocated_openRingIndices = [j for i,j in enumerate(unlocated_openRingIndices) if i not in located_in_this_cp]
#        unlocated_openRingIndices = [unlocated_openRingIndices[k] for k in range(len(unlocated_openRingIndices)) if k not in located_in_this_cp]
    assert not unlocated_openRingIndices
    basic_output_on.dprint("\rFinished locating open rings.  Last cp: %s | Total Elapsed: %s"%(format_time(current_time()-cur_time),format_time(current_time()-start_time)))


#    ###DEBUG ONLY TEST slideshow (of which rings are put in which closed ring pairs)
#    basic_output_on.dprint("creating slideshow of which rings are located between which closed ring pairs...",'nr')
#    from os import path as os_path
#    from options4rings import outputFolder
#    from andysSVGpathTools import svgSlideShow
#    save_dir = os_path.join(outputFolder,'debug','slideshow_closed_pair_inclusions')
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

    #sort the open rings inside each pair of closed rings
    start_time = current_time()
    
    ordering_matrices_pickle_extant = False
    if look4ordering_matrices:
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
            return ring1_isoutside_ring2_cmp(ring_list[idx1], ring_list[idx2], outside_point, bdry_ring.path)
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
            flag_count +=1
            pass
        num_seg_pairs_checked += sum([len(ring_list[i].path)*(len(ring_list[j].path)-1)/2 for (i,j) in combinations(cp.contents,2)])
        try: #lazy fix for test cases where num_seg_pairs2check==0
            percent_complete = num_seg_pairs_checked/num_seg_pairs2check
        except ZeroDivisionError:
            percent_complete = k/len(closed_pairs)
            pass

        if not flag_count:
            psorting = topo_sorted(cp.contents,ring_index_cmp,ordering_matrix=om)

            cp.contents = [cp.contents[index] for index in flattenList(psorting)]
            cp.contents_psorting = psorting
        et_tmp = current_time() - start_time_cp_sorting
        
        if et_tmp > et + 3:
            et=et_tmp
            etr = (1-percent_complete)*et/percent_complete
            basic_output_on.dprint("%s percent complete. Time Elapsed = %s | ETR = %s"%(int(percent_complete*100),format_time(et),format_time(etr)))

    #Output problem cases for manual sorting
    from options4rings import outputFolder
    from os import path as os_path
    from andysmod import output2file
    manual_sort_csvfile = os_path.join(outputFolder,"interactive_sorting",ring_list[0].svgname,"manual_comparisons.csv")
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

        raise Exception("There are %s rings pairs that need to be manually sorted.  Please set 'look4ordering_matrices=True' and run this svg again.  Note: When you run again, there will be an interactive interface to help you sort, but it may be easier to manually enter the needed comparisons in\n%s"%(flag_count,manual_sort_csvfile))
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