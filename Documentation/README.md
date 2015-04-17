## Haar Cascade Training Data

We created our own haar cascade to detect full bodies. The data used to train 
can be founnd at the following URL:

http://www.lookingatpeople.com/download-daimler-ped-det-benchmark/index.html

## Libraries

We are using OpenCV 2.4.10 with Python 2.7. All Python code was ran and tested 
using Ubuntu 14.04 LTS.

## Programs

### Python Programs

All Python code is located in the src/ directory.

#### DensityFlow.py

Displays a density map over the video.

**To run:**
python DensityFlow.py [OPTIONS]

**For help:**
python DensityFlow.py --help

#### CrowdTracking.py

This code handles crowd tracking using Lucas-Kanade. The program runs 
through the video twice. In the first run through the program simply tracks
feature points using Lucas-Kanade. After it has ran through the video it
goes through the recorded tracks and tries to remove invalid tracks.

After the program has all the cleaned up tracks it will display all the 
tracks and replay the video showing the recorded tracks overlaid with the
video.

**To run:**
python CrowdTracking.py [OPTIONS]

**For help:**
python CrowdTracking.py --help

#### OBJCrowdTracking.py

This code handles crowd tracking using Haar Cascades and object detection.

**To run:**
python OBJCrowdTracking.py [OPTIONS]

**For help:**
python OBJCrowdTracking.py --help

#### Rectification.py

The algorithm first takes two pairs of parallel lines calculates the Affine
rectification. This takes the perspective image to an affine image.

Next the algorithm takes two pairs of orthogonal lines calculates Metric
rectification. This takes the affine image to a metric image.

The user is then given a display of the final rectified image where they can
scale, rotate, and translate the results.

Finally, the resulting matrix is printed to the screen where the user can
copy it for uses in other programs.

**To run:**
python Rectification.py [OPTIONS]

**For help:**
python Rectification.py --help

### Unity Program

There is also a 3D player that takes outputed tracks from either 
CrowdTracking.py or OBJCrowdTracking.py and displays the recorded data in a 
3D enviroment.

LittleWalkingPeople.exe

Note this program was ran and tested on Windows 7 with Unity 5 installed.

#### Instructions

When running LittleWalkingPeople.exe you will first be asked what file you wish 
to load. After you select the file the program will begin playing.

You are able to move the camera around using the mouse and keyboard.

Mouse: Look
w: move forward
s: move backward
a: straf left
d: straf right
q: move up
e: move down

You can restart the animation at any time by pressing the space bar. If you wish to 
quit the application simply press the escape key.