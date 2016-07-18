from __future__ import division
from operator import itemgetter
from os import getcwd as os_getcwd
import numpy as np
from itertools import combinations


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def rgb2hex(rgb):
    return ('#%02x%02x%02x' % rgb).upper()


def poly_roots(p, condition=lambda r: True, realonly=False):
    """
    Returns the roots of a polynomial with coefficients given in p.
      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    INPUT:
    p - Rank-1 array-like object of polynomial coefficients.
    realroots - a boolean.  If true, only real roots will be returned.
    condition - a boolean-valued function.  Only roots satisfying this will be
        returned.  If realroots==True, these conditions should assume the roots
        are real.
    OUTPUT: A list containing the roots of the polynomial.
    NOTE:  This uses np.isclose and np.roots
    """
    roots = np.roots(p)
    if realonly:
        roots = [r.real for r in roots if np.isclose(r.imag,0)]
    roots = [r for r in roots if condition(r)]

    duplicates=[]
    for idx,(r1,r2) in enumerate(combinations(roots,2)):
        if np.isclose(r1,r2): #equivalent to abs(r1 - r2) <= (atol + rtol * abs(r2))
           duplicates.append(idx)
    return [r for idx,r in enumerate(roots) if idx not in duplicates]

def output2file(string2output,filename=os_getcwd()+'tempfile_from_output2file',mode=None):
    if mode==None:
        raise Exception("Please give a mode argument when calling this function (use mode='w' to overwrite file or mode='a' to append to file).")
    with open(filename,mode) as fout:
        fout.write(string2output)
def cd__():#change current working directory to be one folder up from current location
    from os import getcwd, chdir
    chdir(getcwd()[0:getcwd().rfind('\\')])  ###WINDOWS ONLY
def cd(newdir):#acts like the most basic use of the unix command
    from os import chdir
    chdir(newdir)
def argmin(somelist):
    return min(enumerate(somelist),key=itemgetter(1))
def ifelse(some_boolean,return_if_true,return_if_false):
    if some_boolean:
        return return_if_true
    else:
        return return_if_false
def n_choose_k(n,k):
    from math import factorial
    return factorial(n)/factorial(k)/factorial(n-k)
def curvature(func,tval,dt=0.01,num_pts_2use=100):
    n = num_pts_2use//2
    ts = [tval+k*dt for k in range(-n,n)]
    a = np.array([[func(t).real,func(t).imag] for t in ts])
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return curvature[n]

def printvars(*varnames): #prints a pasteable "var = var" to paste into your code
#INPUT: variable names as strings
    for varname in varnames:
        print("print('%s = ' + str(%s))"%(varname,varname))
def nparray2nestedlist(npa):#2d arrays only
    return [list(row) for row in npa]
def format001(digits,num):
    if digits<len(str(num)):
        raise Exception("digits<len(str(num))")
    return '0'*(digits-len(str(num))) + str(num)
def format__1(digits,num):
    if digits<len(str(num)):
        raise Exception("digits<len(str(num))")
    return ' '*(digits-len(str(num))) + str(num)
def printmat(arr,row_labels=[], col_labels=[]): #pretty print a matrix (nested list or 2d numpy array)
    try: flattenList(arr)
    except TypeError: arr = [[x] for x in arr] #adds support for vectors
    finally: max_chars = max([len(str(item)) for item in flattenList(arr)+col_labels]) #the maximum number of chars required to display any item in list
    if row_labels==[] and col_labels==[]:
        for row in arr:
            print '[%s]' %(' '.join(format__1(max_chars,i) for i in row))
    elif row_labels!=[] and col_labels!=[]:
        rw = max([len(str(item)) for item in row_labels]) #max char width of row__labels
        print '%s %s' % (' '*(rw+1), ' '.join(format__1(max_chars,i) for i in col_labels))
        for row_label, row in zip(row_labels, arr):
            print '%s [%s]' % (format__1(rw,row_label), ' '.join(format__1(max_chars,i) for i in row))
    else:
        raise Exception("This case is not implemented...either both row_labels and col_labels must be given or neither.")

def eucdist_numpy(l1,l2): #takes in two lists (or tuples) and returns the euclidian distance between them
    from numpy import array
    from numpy.linalg import norm
    return norm(array(list(l1))-array(list(l2)))
def eucnorm_numpy(lon):
    from numpy import array
    from numpy.linalg import norm
    return norm(array(list(lon)))
def eucnormalize_numpy(lon):
    from numpy import array
    from numpy.linalg import norm
    lona = array(list(lon))
    return list(lona/norm(lona))

class Radius(object):
    def __init__(self,origin):
        self.origin = origin

    def __repr__(self):
        return '<Radius object for measuring distance from origin = %s>' %self.origin

    def of(self,pt):
        return abs(pt - self.origin)

def eucdot(l1,l2):
    assert len(l1)==len(l2)
    return sum((l1[i]*l2[i] for i in range(len(l1))))


def bool2bin(boolval): #this can also be done by using True.real
    if boolval:
        return 1
    else:
        return 0

def plotPoints(points):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_points = [z.real for z in points]
    y_points = [z.imag for z in points]
    ax.plot(x_points, y_points, 'b')
    ax.set_xlabel('x-points')
    ax.set_ylabel('y-points')
    ax.set_title('Simple XY point plot')
    fig.show()

def format_time(et):
    if et < 60:
        return '%.1f sec'%et
    elif et < 3600:
        return '%.1f min'%(et/60)
    else:
        return '%.1f hrs'%(et/3600)
import time
from sys import stdout
class Timer(object):
    def __init__(self, name='' ,overwrite=False,formatted=True):
        self.name = name
        self.overwrite = overwrite
        self.formatted = formatted
        if self.name:
            if overwrite:
                stdout.write('\r[%s] Running... '%self.name)
            else:
                stdout.write('[%s] Running... '%self.name)

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self.tstart
        if self.formatted:
            stdout.write('Done (in %s)'%format_time(elapsed_time))
        else:
            stdout.write('Done (in %s seconds)'%elapsed_time)
        stdout.write(ifelse(self.overwrite,'','\n'))
        return

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments."""
    class MemoDict(dict):
        def __init__(self, f_):
            self.f = f_

        def __call__(self, *args):
            return self[args]

        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return MemoDict(f)

def tic(*something_to_say):
    if something_to_say != tuple() and isinstance(something_to_say[0],str):
        print(something_to_say[0])
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(*something_to_say):
    t = time.time()
    if 'startTime_for_tictoc' in globals():
        if something_to_say != tuple():
            print(something_to_say)
        print("Elapsed time is " + str(t - globals()['startTime_for_tictoc']) + " seconds.")
        globals()['startTime_for_tictoc'] = time.time()
    else:
        print("Toc: start time not set")

def pressEnterToContinue(*something_to_say):
    if something_to_say != tuple():
        try: input = raw_input
        except NameError: pass
        input("Press Enter to continue." + " [Note: " + str(something_to_say[0]) + "]")
    else:
        try: input = raw_input
        except NameError: pass
        input("Press Enter to continue.")
    print("Continuing...")
def ignoreCase(*something_to_say):
    try: input = raw_input
    except NameError: pass
    dec = input("Press 'i' to ignore or 'r' to remember (or 'exit' to exit): ")
    if dec=='i':
        return True
    elif dec== 'r':
        return False
    elif dec== 'exit':
        raise Exception("User-forced exit.")
    else:
        ignoreCase(something_to_say)
        
        
def inputyn():
    try: input = raw_input
    except NameError: pass
    dec = input("Enter 'y' or 'n' (or 'e' to exit): ")
    if dec=='y':
        return True
    elif dec== 'n':
        return False
    elif dec== 'e':
        raise Exception("User-forced exit.")
    else:
        inputyn()
        
        
def ask_user(options=None):
    """options should be input as dict whose entries are descriptions.
    Note: q is used (by default) for manual termination."""
    if not options:
        return inputyn()
    try: input = raw_input
    except NameError: pass
    print "Enter one of the following options."
    for key in options.keys():
        print key, ":", dict[key]
    if not options.haskey('q'):
        print "q : exit this menu"
    dec = input()
    print ""
    
    if options.haskey(dec):
        return dec
    elif dec == 'q':
        raise Exception("User-forced exit.")
    else:
        ask_user()


class boolset(list):
    def __contains__(self, element):
        for x in self:
            if x==element:
                return True
        return False

    def booladd(self,element):
        if element not in self:
            self.append(element)

# def extractSVGpathStrings(SVGfileLocation):
    # from completeRingFcns import polylineStr2pathStr
    # from xml.dom import minidom
    # doc = minidom.parse(SVGfileLocation)
    #Use minidom to extract path strings from input SVG
    # path_strings = [(p.getAttribute('d'),p.getAttribute('stroke'), p.parentNode.getAttribute('id')) for p in doc.getElementsByTagName('path')]
    #Use minidom to extract polyline strings from input SVG, convert to path strings, add to list
    # path_strings += [(polylineStr2pathStr(p.getAttribute('points')),p.getAttribute('stroke'), p.parentNode.getAttribute('id')) for p in doc.getElementsByTagName('polyline')]
    #Use minidom to extract line strings from input SVG, convert to path strings, and add them to list
    # path_strings += [('M' + p.getAttribute('x1') + ' ' +p.getAttribute('y1') + 'L'+p.getAttribute('x2') + ' ' + p.getAttribute('y2'),p.getAttribute('stroke'), p.parentNode.getAttribute('id')) for p in doc.getElementsByTagName('line')]
    # return path_strings

class OutputBoolean(object):
    def __init__(self,b):
        self.b = b
    def __repr__(self):
        return str(self.b)

    def dprint(self,s,*nr):
        if self.b:
            if nr == ('nr',):
                from sys import stdout
                stdout.write(s) #does not end the line after printing.
            else:
                print(s)


# def bubbleSortUsingHelper(comparison_fcn,original_itemslist):
# ''' this sorts the itemslist by least to greatest (returns nothing... see note below)
 # fcn(a,b) should return true if a < b
 # Note: this sorts the list in memory without making a copy...
 # to make a copy, change next line to itemslist = original_itemslist.deepcopy()'''
    # itemslist = original_itemslist
    # unsorted = True
    # loopCounter = 0
    # N = len(itemslist)
    # while unsorted and loopCounter<N**3:
        # moreSorted = False
        # for i in range(1,N):
            # if comparison_fcn(itemslist[i],itemslist[i-1]):
                # tmp = itemslist[i]
                # itemslist[i] = itemslist[i-1]
                # itemslist[i-1] = tmp
                # moreSorted = True
        # loopCounter += 1
        # if loopCounter >= N^3-1:
            # raise Exception("loopCounter exceeded limit in bubbleSortUsingHelper...something is wrong with this fcn")
        # if not moreSorted:
            # unsorted = False
            # return ## returns nothing, see note above
def flattenList(list_2_flatten):
    return [item for sublist in list_2_flatten for item in sublist]


from os import path as os_path, getcwd
def open_in_browser(file_location):
    """Attempt to open file_location in the default web browser."""
    # if just the name of the file was given, check if it's in the Current Working Directory.
    if not os_path.isfile(file_location):
        file_location = os_path.join(getcwd(), file_location)
    if not os_path.isfile(file_location):
        raise IOError("\n\nFile not found.")
    try:
        import webbrowser
        new = 2  # open in a new tab, if possible
        webbrowser.get().open(file_location, new=new)

    except ImportError:
        from warnings import warn
        mes = "\nUnable to import webbrowser module.  disvg() fuction will be unable to open created \
              svg files in web browser automatically.\n"
        warn(mes)


class _Getch:#Gets a single character from standard input.  Does not echo to the screen. (from: https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user)
    def __init__(self):
        try:self.impl = _GetchWindows()
        except ImportError:self.impl = _GetchUnix()
    def __call__(self): return self.impl()
class _GetchUnix:
    def __init__(self):
        import tty, sys
    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
class _GetchWindows:
    def __init__(self):
        import msvcrt
    def __call__(self):
        import msvcrt
        return msvcrt.getch()
getch = _Getch()

def createOrderingMatrix(list_of_objects, cmp_fcn, test_symmetry=False): #creates directed graph describing ordering.
    from numpy import zeros
    res = zeros((len(list_of_objects),len(list_of_objects)))
    for i in range(len(list_of_objects)):
        for j in range(i+1,len(list_of_objects)): #this should always be symmetric, so to speed up could just use range(i+1:len(list_of_objects))
            res[i,j] = cmp_fcn(list_of_objects[i], list_of_objects[j])
            res[j,i] = -res[i,j]

    if test_symmetry:
        for i in range(len(list_of_objects)):
            for j in range(i+1,len(list_of_objects)):
                resji = cmp_fcn(list_of_objects[j], list_of_objects[i])
                if res[i,j]!=-resji:
                    raise Exception("Contradiction given by comparison_function: cmp[%s,%s]!=-cmp[%s,%s]"%(i,j,j,i))
    return res
def createDependencyDictionary(ordering_matrix):#create dictionary of dependencies for toposort
    dep_dict = dict()
    for i in range(len(ordering_matrix)):
        dep_dict.update({i:{j for j in range(len(ordering_matrix)) if ordering_matrix[i][j]>0}})
    return dep_dict
def topo_sorted(list_of_objects, cmp_fcn,test_symmetry=False,ordering_matrix=None): #easy2use version of toposort
    from toposort import toposort
    if ordering_matrix is None:
        ordering_matrix = createOrderingMatrix(list_of_objects, cmp_fcn, test_symmetry=test_symmetry)
    dep_dict = createDependencyDictionary(ordering_matrix)
    return toposort(dep_dict)

#def sort_using_toposort(list_of_objects, cmp_fcn): #I THINK THIS FUNCTION WAS NEVER FINISHED AND CAN BE DELETED
#    from numpy import zeros
#    topo_sort_groups, ordMat = topo_sorted(list_of_objects, cmp_fcn)
#    class sort_obect(object):
#        def __init__(ob,index):
#            self.ob = ob; self.id = index, self.isGreaterThan = set(); self.isLessThan = set()
#    sob_list = [sort_object(ob,index) for index,ob in enumerate(list_of_objects)]
#    def record_case(x,y,res,already_updated=zeros((len(list_of_objects),len(list_of_objects)))):
#        if res > 0: #x>y
#            (x,y) = (y,x) #swap x and y so that x<y
#        elif res == 0:
#            return #do nothing
#        ordMat[x.id,y.id] = -1; ordMat[y.id,x.id] = 1
#        for i in range(ordMat.shape[0]):
#            if ordMat[y,i] < 0: #if y<i
#                ordMat[x,i] = -1; ordMat[i,x] = 1
#                if not already_updated[x,i]:
#                    record_case(x,i,-1,already_updated=already_updated)
#            if ordMat[i,x] < 0: #if i<x
#                ordMat[i,y] = -1; ordMat[y,i] = 1
#                if not already_updated[x,i]:
#                    record_case(i,y,-1,already_updated=already_updated)
#    def cmp_fcn_plus(x,y):
#        if ordMat[(x,y)] not in [-1,1]:
#            res = cmp_fcn(x.ob,y.ob)
#            record_case(x,y,res)
#        return ordMat[(x,y)]
#    new_cmp_fcn = cmp_fcn
#    for group in topo_sort_groups:
#        sublist = [sort_object_list[i] for i in group]
#        new_cmp_fcn = gen_cmp_fcn
#        sorted_subllist = sorted(sublist,cmp=cmp_fcn_plus)
#        for sort_object in sorted_sublist:
#            asdfasdfasdf=1 #gonna write my onw I guess
#    return


#def andysort(list_of_objects, cmp_fcn, test_symmetry=True): #creates directed graph describing ordering.
#    from numpy import array as np_array, zeros_like
#
#    #use cmp_fcn to create initial matrix giving sorting
#    ordMat = np_array([[None]*len(list_of_objects)]*len(list_of_objects))
#    for i in range(len(list_of_objects)):
#        for j in range(len(list_of_objects)): #this should always be symmetric, so to speed up could just use range(i+1:len(list_of_objects))
#            ordMat[i,j] = cmp_fcn(list_of_objects[i], list_of_objects[j])
#
#    #test ordMat is skew-symmetric
#    if test_symmetry:
#        for i in range(len(list_of_objects)):
#            for j in range(i,len(list_of_objects)):
#                if ordMat[i,j]!=-ordMat[j,i]:
#                    raise Exception("ordMat[%s,%s]!=ordMat[%s,%s]"%(i,j,j,i))
#
#    #now fill in ordMat by brute force
#    def record_case(x,y,already_updated=zeros_like(ordMat)): #x<y
#        if not already_updated[x,y]:
#            already_updated[x,y] = True
#            ordMat[x,y] = -1; ordMat[y,x] = 1
#            for i in range(ordMat.shape[0]):
#                if ordMat[y,i] < 0: #if y<i
#                    ordMat[x,i] = -1; ordMat[i,x] = 1
#                    if not already_updated[x,i]:
#                        already_updated[x,i] = True
#                        record_case(x,i,already_updated=already_updated)
#                if ordMat[i,x] < 0: #if i<x
#                    record_case(i,y,already_updated=already_updated)
#
#    for i in range(len(list_of_objects)):
#        for j in range(i,len(list_of_objects)):
#            if ordMat[i,j]<0:
#                record_case(i,j)
#            elif ordMat[i,j]>0:
#                record_case(j,i)
#    andy_cmp = lambda x,y: ordMat[x[0],y[0]]
##    printmat(ordMat)
##    printmat(list_of_objects)
#    numbered_list_of_objects = zip(range(len(list_of_objects)),list_of_objects)
#    return [item[1] for item in sorted(numbered_list_of_objects,cmp=andy_cmp)]

def getCVSrow(csv_location, row_num):
    from warnings import warn
    warn("\nUse getcsvrow().\n")
    return getcsvrow(csv_location, row_num)


def getcsvrow(csv_location, row_num):
    with open(csv_location) as f:
        for i, line in enumerate(f):
            if i == row_num:
                return line

def adj(attr): #for use with array_starting_at_one()
    if attr==None:
        return attr
    else:
        return attr-1
def adjust_slice(x): #for use with array_starting_at_one()
    if isinstance(x,int):
        return x-1
    elif isinstance(x,slice):
        return slice(*[adj(attrib) for attrib in (x.start,x.stop,x.step)])
    elif isinstance(x,list):
        return slice(x[0]-1,x[-1]-1,1)
    else:
        raise Exception("Expected slice, list, or int.")
class array_starting_at_one(list): #creates a numpy array with indices starting at 1 (instead of 0)
    def __init__(self,np_array):
        from numpy import array as numpyarray
        self.np_array = numpyarray(np_array)
    def __getitem__(self,i):
        if isinstance(i,int):
            i=i-1
        elif isinstance(i,tuple):
            i = tuple([adjust_slice(x) for x in i])
        else:
            return array_starting_at_one(self.np_array[adjust_slice(x)])
        return self.np_array[i]
    def __setitem__(self,i,y):
        if isinstance(i,int):
            self.np_array[i-1] = y
        elif isinstance(i,tuple):
            self.np_array[tuple([adjust_slice(x) for x in i])] = y
        else:
            self.np_array[adjust_slice(x)] = y
    def __getslice__(self,i,j):
        return array_starting_at_one(self.np_array[(i-1):(j-1)])
    def __setslice__(self,i,j,y):
        self.np_array[i-1:j-1]=y
    def __repr__(self):
        print self.np_array
    def __str__(self):
        return str(self.np_array)

from threading import Timer as threading_Timer
class RepeatedTimer(object): #from 'https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds'
#Usage example:
#from time import sleep
#def hello(name):
#    print "Hello %s!" % name
#
#print "starting..."
#rt = RepeatedTimer(1, hello, "World") # it auto-starts, no need of rt.start()
#try:
#    sleep(5) # your long-running job goes here...
#finally:
#    rt.stop() # better in a try/finally block to make sure the program ends!
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()
    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)
    def start(self):
        if not self.is_running:
            self._timer = threading_Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True
    def stop(self):
        self._timer.cancel()
        self.is_running = False
def cv2hist(img):
    import cv2
    from matplotlib import pyplot as plt
    color = ('b','g','r')
    for i,col in enumerate(color):
        plt.subplot(3,1,i+1);plt.hold(False)
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
def id(x):#https://ipython-books.github.io/featured-01/
    # This function returns the memory block address of an array.
    return x.__array_interface__['data'][0]
def limit(func, t0, side=0, epsilon=1e-16, delta0=0.5, maxits=10000, n=5):
    """computes the limit of func(t) as t->t0
    Note: The domain of func is assumed to be (t0-delta0,t0),(t0,t0+delta0), or
        the union of these intervals depending on side
    Note: the function will possibly be evaluated n*maxits times
    INPUT:
    side - determines whether right (side>0) or left (side<0); two sided if 0
    delta0 is the initial delta
    """
    from random import uniform
    assert epsilon > 0 and delta0 > 0 and maxits > 0

    if side > 0:
        delta = float(delta0)
    elif side < 0:
        delta = -float(delta0)
    else:
        posres = limit(func, t0, side=1, epsilon=epsilon, maxits=maxits, delta0=delta0)
        negres = limit(func, t0, side=-1, epsilon=epsilon, maxits=maxits, delta0=delta0)
        if abs(posres - negres) <= 2 * epsilon:
            return (posres + negres) / 2
        else:
            raise Exception("\n%s = the left side limit != the right side limit = %s." % (negres, posres))
    lim = epsilon * 10
    old_lim = -epsilon * 10
    its = 0
    while abs(lim - old_lim) >= epsilon and its < maxits:
        its += 1
        old_lim = lim
        ts = [uniform(t0, t0 + delta) for dummy in range(n)]
        lims = map(func, ts)
        lim = sum(lims) / len(lims)
        delta /= n
    if its >= maxits:
        raise Exception("Maximum iterations reached.")
    return lim


#def timeme(func,arglist,runs_per_loop=1000,use_best=3,label=None):
#	raise Exception("This method seems to not work... use timeit")
#    def is_sequence(obj):
#        return hasattr(obj, '__len__') and hasattr(obj, '__getitem__')
#    for k,arg in enumerate(arglist):
#        if not is_sequence(arg):
#            arglist[k] = [arg]
#
#    numloops = len(arglist)//runs_per_loop
#    if not numloops:
#        new_runs_per_loop = len(arglist)
#        timeme(func, arglist, runs_per_loop=new_runs_per_loop,
#               use_best=use_best, label=label)
#        return
#
#    from time import time
#    c=[]
#    for k in range(numloops):
#        a = time()
#        for j in range(runs_per_loop):
#            func(*arglist[k])
#        b = time()
#        c.append(b-a)
#
#    if label==None:
#        try:
#            label = func.__name__
#        except AttributeError:
#            label = "unlabeled"
#
#    best_used = min(len(c),use_best)
#    stats = (numloops,runs_per_loop,best_used,sum(sorted(c)[0:best_used]))
#    mes = label + " : "
#    mes += "After %s loops of %s, the sum of the best %s loop-times was %s s\
#        "%stats
#    print mes

def regress(x_data, y_data, title='Untitled Plot', xlabel='x-axis', ylabel='y-axis', connect_dots=False):
    """Linear Regression for 1d data."""
    import matplotlib.pyplot as plt

    # sort the data
    reorder = sorted(range(len(x_data)), key = lambda ii: x_data[ii])
    x_data = [x_data[ii] for ii in reorder]
    y_data = [y_data[ii] for ii in reorder]

    #Plot the data from y_points
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    plt.scatter(x_data, y_data, s=30, alpha=0.15, marker='o')  # draw circles around points
    plt.plot(x_data, y_data, 'bo')
    if connect_dots:
        ax.plot(x_data, y_data, 'b')

    # determine best fit line
    bfl = np.polyfit(x_data, y_data, 1, full=True)

    slope=bfl[0][0]
    intercept=bfl[0][1]
    xl = [min(x_data), max(x_data)]
    yl = [slope*xx + intercept  for xx in xl]

    # coefficient of determination
    variance = np.var(y_data)
    residuals = np.var([(slope*xx + intercept - yy) for xx, yy in zip(x_data,y_data)])
    Rsqr = np.round(1 - residuals/variance, decimals=2)

    # Draw labels and text
    plt.text(.8*max(x_data)+.2*min(x_data),.8*max(y_data)+.2*min(y_data),'$R^2 = %0.4f$'% Rsqr, fontsize=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # error bounds
    yerr = [abs(slope*xx + intercept - yy) for xx, yy in zip(x_data,y_data)]
    par = np.polyfit(x_data, yerr, 2, full=True)

    yerrUpper = [(xx*slope + intercept) + (par[0][0]*xx**2 + par[0][1]*xx + par[0][2])
                 for xx, yy in zip(x_data,y_data)]
    yerrLower = [(xx*slope + intercept) - (par[0][0]*xx**2 + par[0][1]*xx + par[0][2])
                 for xx, yy in zip(x_data,y_data)]

    plt.plot(xl, yl, '-r')
    plt.plot(x_data, yerrLower, '--r')
    plt.plot(x_data, yerrUpper, '--r')
    plt.show()
    return slope, intercept, Rsqr


import datetime
def daysofyear(year):
    """Returns a sorted list of date objects, one for each day of the year. """
    daysofyear = []
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                daysofyear.append(datetime.date(year, month, day))
            except ValueError:
                break


def date2dayofyear(date):
    """Returns the day of the year, an integer, d, 1 <= d <= 366 (or whatever
        the maximum number of days in a year on the Gregorian calendar is)."""
    if isinstance(date, datetime.date):
        year = date.year
        day = date
    else:  # assume formatted yyyy-mm-dd
        year = int(date[:4])
        day = datetime.strptime(d2, "%Y-%m-%d")
    firstdayofyear = datetime.date(year, 1, 1)
    diff = (day - firstdayofyear).days
    return diff + 1