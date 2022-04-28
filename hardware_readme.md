# Hardware Report





__Key Components__

___OAK-D___
The OAK-D is the primary component of Opticle that facilitates 3 major functions through its 4K RGB Camera for visual perception, a stereo pair for depth perception and an Intel Myriad X Visual processing Unit to function as the brain/processor capable of running modern neural networks while simultaneously creating a depth map from the stereo pair of images in real time.

__Hardware Specifications:__
The OAK camera uses USB-C cable for communication and power. It supports both USB2 and USB3(5Gbps / 10 Gbps)
| Camera Specs | Color Camera | Stereo Pair|
| ------ | ------ | ------|
| Sensor | IMX378 |OV9282
| DFOV / HFOV / VFOV | 81° / 69° / 55° | 82° / 72° / 50°
| Resolution | 12MP (4032x3040) |1MP (1280x800)
| Max Frame Rate | 60 FPS | 120 FPS|
| Pixel Size |  1.55µm x 1.55µm| 3µm x 3µm
__IMAGE HERE__

___Raspberry Pi 4B___
The Raspberry Pi 4B is the brain of the device that connects all the different components together. Firstly, with 2 USB ports, it is connected to a 10000maH power bank (5V) and the OAK-D camera to receive the video feed and produce actionable feedback such as haptic and auditory output. 

__IMAGE HERE__

___Wrist Mount___
While the OAK-D, Power Source and the Raspberry Pi 4B will be on the chest mount that the user be equipped with, there is also a 2nd wrist mount component that the user will wear like a watch. The wrist mount is equipped with a LRA motor that sits on the base of the mount and a raspberry pi zero that sits in the hollow region between the roof and the base. The raspberry pi zero will interface with the raspberry pi 4B where the LRA motor will vibrate if an object is present inside the point cloud region of 2mx1mx1.7m.

IMAGE HERE

Select components were constructed on OnShape(CAD Software) and the files are attached below in the case that the reader requires personal customization.

| Plugin | README |
| ------ | ------ |
| Wrist Mount | LINK |
| Pi Holder + Switch | LINK|
| Power Holder | LINK |
