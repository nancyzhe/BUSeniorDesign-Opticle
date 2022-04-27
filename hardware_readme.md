# Hardware Report

## _Components Overview_

__Key Components__

# ___OAK-D___

The OAK-D is the primary component of Opticle that facilitates 3 major functions through its 4K RGB Camera for visual perception, a stereo pair for depth perception and an Intel Myriad X Visual processing Unit to function as the brain/processor capable of running modern neural networks while simultaneously creating a depth map from the stereo pair of images in real time.


IMAGE HERE

___Raspberry Pi 4B___
The Raspberry Pi 4B is the brain of the device that connects all the different components together. Firstly, with 2 USB ports, it is connected to a 10000maH power bank (5V) and the OAK-D camera to receive the video feed and produce actionable feedback such as haptic and auditory output. 

IMAGE HERE

___Wrist Mount___
While the OAK-D, power source and the Raspberry Pi 4B will be on the chest mount that the user will wear, the 2nd component will be the wrist mount that the user will wear like a watch. The wrist mount is equipped with a LRA motor that sits on the base of the mount and a raspberry pi zero that sits in the hollow region between the roof and the base. The raspberry pi zero will interface with the raspberry pi 4B where the LRA motor will vibrate if an object is present inside the point cloud region of 2mx1mx1.7m.

IMAGE HERE
