SVGTree
============

SVGTree is open-source dendrochronology software which takes in an SVG sketch
of tree rings and returns the areas of the rings, computer generated transects, and more.

Input Data Format
-----------------
SVGTree can take in any svg sketch drawn using paths of Bezier curves (and/or lines, polylines, and polygons) on the following two conditions:

1. There must be a closed path surrounding a **blue line**.  The midpoint of this line will be used as the starting point of all transects.

2. There must be a closed path surrounding all other paths.

In other words, the inner-most ring and the outer-most ring must be closed 
loops and there should be a blue line inside the inner-most ring to mark where
the transects will start.

To Run
------
1. Follow the instructions below to install any prerequisites needed.

2. Download and unzip SVGTree.

3. Move some SVG files you want to extract information from to the input folder (found inside the SVGTree folder that you just unzipped).

4. Open a terminal/command-prompt, navigate into the SVGTree folder and enter the following command (without the $).

$ python svgtree.py

This will extract data from any SVG files in the `.../SVGTree/input/` folder and store output (by default, areas, and 50 evenly space transects) in the `.../SVGTree/output/` folder.


Options
-------
**To increase/decrease the number of transects found**, use the -n <number> flag

$ python svgtree.py -n 500

**To turn off area calculations**, use the -a flag

$ python svgtree.py -n 500 -a

**To specify the input/output directories:**

$ python svgtree.py -i "your_svg_file.svg" -o "your_desired_output_directory"

**For more basic options**, enter the command:

$ python svgtree.py -h

**For advanced options**, read through the file "options4rings.py".

Prerequisites
-------------
-  **python 2.x**
-  **numpy**
-  **scipy**
-  **svgwrite**
-  **svgpathtools**

Setup
-----

1. Get Python 2.  

Note: If you have a **Mac** or are running **Linux**, you already have Python 2.x.  If you're on **Windows**, go download Python 2 and install it.

2. Install the necessary python packages. 

Note: This is easy using pip (which typically comes with Python).  Just open up a terminal/command-prompt and enter the following four commands (without the $).

$ pip install numpy

$ pip install scipy

$ pip install svgwrite

$ pip install svgpathtools

For help
--------
Contact me, AndyAPort@gmail.com

Licence
-------

This module is under a MIT License.
