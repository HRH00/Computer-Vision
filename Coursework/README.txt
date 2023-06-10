The training data should be in the Datastore directory, Place the data set supplied in the labs
in the /Datastore/Supplied_Data directory, with all videos of the same label class in a directory
of their own - Titled with the label i.e all walking images in /Datastore/Supplied_Data/Walking. 

Make sure there are no empty directories which are not a part of the data set in /Datastore/Supplied_Data,
Other non .avi files should not interfer with data collection (hopefully).


The version of python needed to run this code is 3.11
The program has only been tested on a windows 11 machine.

The python packages imported in each file must be installed:
pip install sklearn
pip install 
pip install opencv-contrib-python
pip install numpy



os, pickle, sys, time modules should already be preinstalled in the python libraries. 

A package of helper functions has been made, package_901476 - make sure this package directory is placed 
inside the same directory that the indivudual scripts are called from.