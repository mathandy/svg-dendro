from __future__ import division, print_function
import numpy as np
import os
import time
from sys import stdout


def output2file(string2output, 
                filename=os.getcwd()+'tempfile_from_output2file', 
                mode=None):
    if mode is None:
        raise Exception(
            "Please give a mode argument when calling this function "
            "(use mode='w' to overwrite file or mode='a' to append to file)."
        )
    
    with open(filename, mode) as fout:
        fout.write(string2output)


def n_choose_k(n, k):
    from math import factorial
    return factorial(n) / factorial(k) / factorial(n - k)


def curvature(func, tval, dt=0.01, num_pts_2use=100):
    n = num_pts_2use // 2
    ts = [tval + k * dt for k in range(-n, n)]
    a = np.array([[func(t).real, func(t).imag] for t in ts])
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    denom = (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / denom
    return curvature[n]


def printvars(*vargs):
    """prints a pasteable "var = var" to paste into your code

    Args:
        *vargs: variable names as strings

    Returns:
        None
    """
    for varname in vargs:
        print("print('%s = ' + str(%s))" % (varname, varname))


def nparray2nestedlist(npa):  # 2d arrays only
    return [list(row) for row in npa]


def format001(digits, num):
    if digits < len(str(num)):
        raise Exception("digits<len(str(num))")
    return '0' * (digits - len(str(num))) + str(num)


def format__1(digits, num):
    if digits < len(str(num)):
        raise Exception("digits<len(str(num))")
    return ' ' * (digits - len(str(num))) + str(num)


def printmat(arr, row_labels=[], col_labels=[]):
    """pretty print a matrix (nested list or 2d numpy array)"""
    try:
        flattenList(arr)
    except TypeError:
        arr = [[x] for x in arr]  # adds support for vectors
    finally:
        # the maximum number of chars required to display any item in list
        max_chars = \
            max([len(str(item)) for item in flattenList(arr) + col_labels])

    if row_labels == [] and col_labels == []:
        for row in arr:
            print('[%s]' % (' '.join(format__1(max_chars, i) for i in row)))
    elif row_labels != [] and col_labels != []:

        # max char width of row__labels
        rw = max([len(str(item)) for item in row_labels])

        print('%s %s' % (' ' * (rw + 1),
                         ' '.join(format__1(max_chars, i) for i in col_labels)))
        for row_label, row in zip(row_labels, arr):
            print('%s [%s]' % (format__1(rw, row_label),
                               ' '.join(format__1(max_chars, i) for i in row)))
    else:
        raise Exception("This case is not implemented...either both "
                        "row_labels and col_labels must be given or neither.")


def eucdist_numpy(l1, l2):
    """euclidian distance between two lists"""
    from numpy import array
    from numpy.linalg import norm
    return norm(array(list(l1)) - array(list(l2)))


def eucnorm_numpy(lon):
    from numpy import array
    from numpy.linalg import norm
    return norm(array(list(lon)))


def eucnormalize_numpy(lon):
    from numpy import array
    from numpy.linalg import norm
    lona = array(list(lon))
    return list(lona / norm(lona))


class Radius(object):
    def __init__(self, origin):
        self.origin = origin

    def __repr__(self):
        return '<Radius object for measuring distance from origin = %s>' % self.origin

    def of(self, pt):
        return abs(pt - self.origin)


def eucdot(l1, l2):
    assert len(l1) == len(l2)
    return sum((l1[i] * l2[i] for i in range(len(l1))))


def bool2bin(boolval):  # this can also be done by using True.real
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
        return '%.1f sec' % et
    elif et < 3600:
        return '%.1f min' % (et / 60)
    else:
        return '%.1f hrs' % (et / 3600)


class Timer(object):
    def __init__(self, name='', overwrite=False, formatted=True):
        self.name = name
        self.overwrite = overwrite
        self.formatted = formatted
        if self.name:
            if overwrite:
                stdout.write('\r[%s] Running... ' % self.name)
            else:
                stdout.write('[%s] Running... ' % self.name)

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self.tstart
        if self.formatted:
            stdout.write('Done (in %s)' % format_time(elapsed_time))
        else:
            stdout.write('Done (in %s seconds)' % elapsed_time)
        stdout.write('' if self.overwrite else '\n')
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


def ignoreCase(*something_to_say):
    dec = input("Press 'i' to ignore or 'r' to remember (or 'exit' to exit): ")
    if dec == 'i':
        return True
    elif dec == 'r':
        return False
    elif dec == 'exit':
        raise Exception("User-forced exit.")
    else:
        ignoreCase(something_to_say)


def inputyn():
    try:
        input = raw_input
    except NameError:
        pass
    dec = input("Enter 'y' or 'n' (or 'e' to exit): ")
    if dec == 'y':
        return True
    elif dec == 'n':
        return False
    elif dec == 'e':
        raise Exception("User-forced exit.")
    else:
        inputyn()


def ask_user(options=None):
    """options should be input as dict whose entries are descriptions.
    Note: q is used (by default) for manual termination."""
    if not options:
        return inputyn()
    try:
        input = raw_input
    except NameError:
        pass
    print("Enter one of the following options.")
    for key in options.keys():
        print(key, ":", dict[key])
    if not options.haskey('q'):
        print("q : exit this menu")
    dec = input()
    print("")

    if options.haskey(dec):
        return dec
    elif dec == 'q':
        raise Exception("User-forced exit.")
    else:
        ask_user()


class boolset(list):
    def __contains__(self, element):
        for x in self:
            if x == element:
                return True
        return False

    def booladd(self, element):
        if element not in self:
            self.append(element)


class ConditionalPrint(object):
    def __init__(self, b):
        self.b = b

    def __repr__(self):
        return str(self.b)

    def __call__(self, s, *nr):
        self.dprint(s, *nr)

    def dprint(self, s, *nr):
        if self.b:
            if nr == ('nr',):
                from sys import stdout
                stdout.write(s)  # does not end the line after printing.
            else:
                print(s)


def flattenList(list_2_flatten):
    return [item for sublist in list_2_flatten for item in sublist]


def open_in_browser(file_location):
    """Attempt to open file_location in the default web browser."""

    # if just the name of the file was given, check if it's in the CWD
    if not os.path.isfile(file_location):
        file_location = os.path.join(os.getcwd(), file_location)
    if not os.path.isfile(file_location):
        raise IOError("\n\nFile not found.")
    try:
        import webbrowser
        new = 2  # open in a new tab, if possible
        webbrowser.get().open(file_location, new=new)

    except ImportError:
        from warnings import warn
        mes = ("\nUnable to import webbrowser module.  disvg() fuction "
               "will be unable to open created svg files in web "
               "browser automatically.\n")
        warn(mes)


class _Getch:
    """Gets a single character from standard input.

    Does not echo to the screen.

    Credit: stackoverflow.com/questions/510357
    """
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


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


def createOrderingMatrix(list_of_objects, cmp_fcn, test_symmetry=False):
    """creates directed graph describing ordering."""
    from numpy import zeros
    res = zeros((len(list_of_objects), len(list_of_objects)))
    for i in range(len(list_of_objects)):
        # this should always be symmetric, so to speed up could just
        # use range(i+1:len(list_of_objects))
        for j in range(i + 1, len(list_of_objects)):
            res[i, j] = cmp_fcn(list_of_objects[i], list_of_objects[j])
            res[j, i] = -res[i, j]

    if test_symmetry:
        for i in range(len(list_of_objects)):
            for j in range(i + 1, len(list_of_objects)):
                resji = cmp_fcn(list_of_objects[j], list_of_objects[i])
                if res[i, j] != -resji:
                    raise Exception(
                        "Contradiction given by comparison_function: "
                        "cmp[%s,%s]!=-cmp[%s,%s]" % (i, j, j, i))
    return res


def createDependencyDictionary(ordering_matrix):
    """create dictionary of dependencies for toposort"""
    dep_dict = dict()
    for i in range(len(ordering_matrix)):
        dep_dict.update({i: {j for j in range(len(ordering_matrix))
                             if ordering_matrix[i][j] > 0}})
    return dep_dict


def topo_sorted(list_of_objects, cmp_fcn, test_symmetry=False,
                ordering_matrix=None):
    """easy-to-use version of toposort"""
    from toposort import toposort
    if ordering_matrix is None:
        ordering_matrix = createOrderingMatrix(
            list_of_objects, cmp_fcn, test_symmetry=test_symmetry)
    dep_dict = createDependencyDictionary(ordering_matrix)
    return toposort(dep_dict)


def limit(func, t0, side=0, epsilon=1e-16, delta0=0.5, maxits=10000, n=5):
    """computes the limit of func(t) as t->t0

    Note: The domain of func is assumed to be (t0-delta0,t0),(t0,t0+delta0), or
        the union of these intervals depending on side
    Note: the function will possibly be evaluated n*maxits times

    Args:
        side: determines whether
            right (side > 0) or left (side < 0) or two-sided (side == 0)
        delta0: is the initial delta

    """
    from random import uniform
    assert epsilon > 0 and delta0 > 0 and maxits > 0

    if side > 0:
        delta = float(delta0)
    elif side < 0:
        delta = -float(delta0)
    else:
        posres = limit(
            func, t0, side=1, epsilon=epsilon, maxits=maxits, delta0=delta0)
        negres = limit(
            func, t0, side=-1, epsilon=epsilon, maxits=maxits, delta0=delta0)

        if abs(posres - negres) <= 2 * epsilon:
            return (posres + negres) / 2
        else:
            raise Exception("\n%s = the left side limit != the right "
                            "side limit = %s." % (negres, posres))
    lim = epsilon * 10
    old_lim = -epsilon * 10
    its = 0
    for its in range(maxits):
        if not abs(lim - old_lim) >= epsilon:
            break
        old_lim = lim
        ts = [uniform(t0, t0 + delta) for _ in range(n)]
        lims = map(func, ts)
        lim = sum(lims) / len(lims)
        delta /= n
    if its >= maxits - 1:
        raise Exception("Maximum iterations reached.")
    return lim
