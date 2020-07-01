SVG Dendro (formerly SVGTree)
=============================

SVG Dendro is open-source dendrochronology software which takes in an SVG sketch
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

2. Download and unzip SVG Dendro.

3. Move some SVG files you want to extract information from to the data folder (found inside the SVG Dendro folder that you just unzipped).

4. Open a terminal/command-prompt, navigate into the SVG Dendro folder and enter the following command (without the $).

$ python main.py -i data/real_examples/

This will extract data from any SVG files in the `<SVG Dendro Folder>/input/examples/real_examples` folder and store output (by default, areas, and 50 evenly space transects) in the `<SVG Dendro Folder>/output` folder.


Options
-------
**To increase/decrease the number of transects found**, use the -n <number> flag

$ python main.py -n 500

**To turn off area calculations**, use the -a flag

$ python main.py -n 500 -a

**To specify the input/output directories:**

$ python main.py -i "your_svg_file.svg" -o "your_desired_output_directory"

**For more basic options**, enter the command:

$ python main.py -h

**For advanced options**, read through the file "options4rings.py".

Prerequisites
-------------
-  **python 3.x**
-  **scipy**
-  **svgpathtools**

Setup
-----

1. Get/install Python 3.

2. Install the necessary python packages.

Note: This is easy using pip (which typically comes with Python).  Just
open up a terminal/command-prompt and enter the following two commands
(without the $).

$ pip install svgpathtools

$ pip install scipy

Test
----
$ python main.py --fakes  # test using the simple fakes in data/test_examples

$ python main.py --reals  # test using the real examples in data/real_examples

For help
--------
Contact me, AndyAPort@gmail.com

Licence
-------

This module is under a MIT License.
