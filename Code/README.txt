The training data should be in the Datastore directory, Place the data set supplied in the labs
in the /Datastore/Supplied_Data directory, with all videos of the same label class in a directory
of their own - Titled with the label i.e all "Walking" images in /Datastore/Supplied_Data/Walking. 

Make sure there are no empty directories which are not a part of the data set in /Datastore/Supplied_Data,

Python Veriosn 3.11 reccomended

The program has only been tested on a windows 11 machine.

The python packages imported in each file must be installed:
pip install scikit-learn
pip install matplotlib
pip install opencv-contrib-python
pip install numpy


Alternativly run 
installPythonPackages.ps1 using powershell to do this for you

os, pickle, sys & time modules should already be accessible in the python libraries. 

A package of helper functions has been made, package_901476 - make sure this package directory is placed 
inside the same directory that the indivudual scripts are called from.

If you change the data set, that path needs to be updated in each "trainer.py" script, ensure you delete the 
old MHI_array.pkl or the program may use previous datasets. The data set must be relativly small in size due
to 'pickle' 10MB file size limitation, data could be cached using other means as an improvement