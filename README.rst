SVGTree
============

SVGTree is open-source dendrochronology software which takes in an SVG sketch
of tree rings and returns the areas of the rings, computer generated transects, and more.

Input Data Format
-----------------
SVGTree can take in any svg sketch drawn using paths of Bezier curves (and/or lines, polylines, and polygons) on the following two conditions:

1) There must be a closed path surrounding a **blue line**.  The midpoint of 
this line will be used as the starting point of all transects.

2) There must be a closed path surrounding all other paths.

In other words, the inner-most ring and the outer-most ring must be closed 
loops and there should be a blue line inside the inner-most ring to mark where
the transects will start.

To Run
------
1) Follow the setup instructions below.
2) Download and then unzip SVGTree.
3) Read through the file "options.py".  This is where you point the code to 
where you're storing the SVGs you'd like to input, and explains all the 
options available.  If you have any questions, feel free to ask me 
(AndyAPort@gmail.com).

4) Open a terminal/command-prompt, navigate into the SVGTree folder and enter 
the following command.
$ python svgtree.py

Prerequisites
-------------
-  **python 2.x**
-  **numpy**
-  **svgwrite**
-  **svgpathtools**

Setup
-----
1) Get Python 2:
If you have a Mac or are running Linux, you already have Python 2.x.
If you're on windows, go download Python 2 and install it.
If not already installed, you can **install the prerequisites** using
pip.

2) Install the necessary python packages. This is easy using pip.  Just open 
up a terminal/command-prompt and enter the following three commands.
$ pip install numpy
$ pip install svgwrite
$ pip install svgpathtools

Licence
-------

This module is under a MIT License.

