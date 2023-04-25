# DSL_Final_Project
# Labeling Armor Plates using YOLO v5 and Training that Model on the Beaglebone AI-64
By Paul Han, Drew Conyers, Michael Routh, Tristan Becnel, Kenneth Pinzon, Isabel Aguilera, Tanya Shiramagond, and Anthony Lam

![https://cdn.discordapp.com/attachments/1095549530583343116/1100186102838607982/Fleet_Render_1x1.png](https://cdn.discordapp.com/attachments/1095549530583343116/1100186102838607982/Fleet_Render_1x1.png)

## Section 1: Problem Overview
The Robomaster competition, an annual robotics contest held by DJI, gathers teams from universities and institutions worldwide to participate in a series of challenges using robotics and computer vision. The main competition at this event is the robot battle, where teams design and build custom robots to compete in an arena style game. Within the competition environment robots navigate through the course, identify and interact with specific objects, and engage in simulated combat with opposing teams’ robots. The game comprises multiple rounds, with each team’s objective being to score points by accurately hitting their opponent’s robots’ armor plates. The armor plates are equipped with lights and numbers, and damage is only registered when these plates are hit. This unique aspect of the competition emphasizes the importance of computer vision and object detection capabilities, as robots must quickly and accurately identify and target the armor plates of their adversaries to score points and ultimately achieve victory.

![https://media.discordapp.net/attachments/1095549530583343116/1100193804864213114/image.png?width=1031&height=662](https://media.discordapp.net/attachments/1095549530583343116/1100193804864213114/image.png?width=1031&height=662)

In this project, we used an object detection machine learning algorithm called YOLO v5 (You Only Look Once) for detecting Robomaster armor plates with high, consistent accuracy. We then enhanced its performance on a System-on-Chip (SoC) called Beaglebone AI-64, equipped with an Intel Realsense camera, to achieve very high frames per second. Our approach involves using ONNX (Open Neural Network Exchange) files, which accelerate matrix operations and inferences for faster calculations. To accomplish this, we collected and labeled images of armor plates attached to competition robots and trained YOLO v5 using these images in a supervised learning manner. Our dataset comprises images provided by Purdue University’s Boilderbots and the University of Texas’ Stampede Robomaster teams, which were labeled using CVAT (Computer Vision Annotation Tool). We used approximately four thousand images for training. The YOLO v5 model, pre-trained on the COCO dataset, is retrained on the Robomaster dataset to detect armor places, a critical component of Robomaster competition, as damage in the game is only incurred when armor places are hit. Using YOLO API, we modified and tailored the YOLO v5 model to suit the requirements of the Robomaster competition and will significantly contribute to the efficiency and effectiveness of robot combat strategies.
![https://cdn.discordapp.com/attachments/1095549530583343116/1100199334504562698/Software.png](https://cdn.discordapp.com/attachments/1095549530583343116/1100199334504562698/Software.png)
## Section 2: YOLO v5 Overview
YOLO v5 is a computer vision model and is used for object detection. It performs well relative to its inference time, so it is known to process images quickly with a high degree of accuracy.
The YOLO v5 network is made of 3 pieces: the backbone, the neck, and the head. The backbone is a convolutional neural network that takes the image input and forms image features by putting it through different layers. The neck merges these image features together to make a prediction. The head forms a class prediction based on the acquired features, which overall creates a sparse prediction of the location of the detected object.
YOLO v5 uses data augmentation and loss calculation techniques as their main training procedures. In our case, the YOLO model took the Robomaster images with labeled armor plates and used these training procedures to learn how to detect the armor plates. 

## Section 3: Data Collection and Preprocessing
Our dataset of images is from the Boilerplates Purdue University Robomaster team and from the UT Robomaster team Stampede. The images that were provided consisted of metal plates that had blue lights and of metal plates that had red lights. We then proceeded to label these images using a publicly available tool called CVAT (Computer Vision Annotation Tool, cvat.ai). We created two distinct labels to distinguish these plates with a red label titled “Red Plate” for plates with red lights and a blue label titled “Blue Plate” for plates with blue lights. Each team member was responsible for labeling ~500 images. Using CVAT, we each labeled 500 plates by drawing squares around the metal plates. If the image was too blurry or both lights on the plate were not visible then we would skip that image. Once we were finished labeling the images we exported the data as YOLO 1.1 format. Once every team member was finished we were now ready to start training and optimizing our model.
![https://media.discordapp.net/attachments/1095549530583343116/1100194213934673990/image.png?width=1375&height=662](https://media.discordapp.net/attachments/1095549530583343116/1100194213934673990/image.png?width=1375&height=662)

## Section 4: Performance Optimization with ONNX
ONNX or Open Neural Network Exchange is an open-source library dedicated to facilitating the exchange of neural networks between different frameworks and tools. The library allows for the expression of different deep learning models to be represented in a common format which allows models to be used across platforms and multiple frameworks. 

Our team is capitalizing on ONNX’s efficiency and model optimization to maximize our YOLO's performance. ONNX supports hardware acceleration on specialized accelerators which boosts stellar performance due to the standardization of model formatting. ONNX also supports quantization, reducing memory usage and computation time as well as having the ability of ‘pruning’ which gets rid of  unnecessary model complexity.

Lastly, although there may be other machine learning model optimization libraries like NVIDIA’s CUDA, our model was made by Texas Instruments and our beaglebone has proprietary TI chips integrated within the SoC. Therefore the decision to use ONNX is obvious. The library is the most compatible with the model, hardware, and can push the most performance out of our project. 

## Section 5: Integrating with Beaglebone AI-64 and Intel Realsense Camera
Integrating the YOLO v5 object detection model with the Beaglebone AI-64 SoC and the Intel Realsense camera involves the following series of steps to optimize the detection of armor plates in the Robomaster competition:

The first step is to initialize and configure the BeagleBone. Booting of Beaglebone is done through an SD card with the latest Debian image available. The boot is done with the image on the SD card, not the eMMC (Embedded MultiMediaCard) on the board. Once first properly booted it should always be launched from the SD card given the card is on the board.

The second step is to install all required firmware. The major package to be installed is [BeagleBoard Device Trees](https://github.com/beagleboard/BeagleBoard-DeviceTrees/). Installation of Python and libraries required for running the YOLO v5 model, such as PyTorch and OpenCV. Additionally, drivers and software for Intel Realsense cameras must be installed to enable communication between the camera and the Beaglebone AI-64 board. The Beaglebone AI-64 board is properly configured with all software dependencies installed. 

The third step is to enable internet access on the BeagleBone. The default gateway is set to the desired ip address. The time date is then configured to the Chicago timezone, Network Time Protocol is disabled, and is set to military time. Set resolv.conf as open by adding  “nameserver 8.8.8.8” to the end of the file. The final step is to connect the BeagleBone to the internet using a USB connected to a Windows computer, [detailed explanation can be found here](https://www.digikey.com/en/maker/blogs/how-to-connect-a-beaglebone-black-to-the-internet-using-usb).

The fourth step is to install Realsense drivers on the Beaglebone. Detailed instructions for the package installation are [found here](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python).

![https://media.discordapp.net/attachments/1095549530583343116/1100207977060909106/Screen_Shot_2023-04-24_at_6.53.36_PM.png?width=1440&height=551](https://media.discordapp.net/attachments/1095549530583343116/1100207977060909106/Screen_Shot_2023-04-24_at_6.53.36_PM.png?width=1440&height=551)

## Section 6: Test and Evaluation
We tested our trained data model on the images provided by the UT Robomaster Stampede team and at first our model was insufficient with improperly labeling plates with red lights. From our initial testing, we lowered the learning rate by a factor of 15 times and this improved our data model. We also lowered the resolution because smaller images have faster inference times. With faster inference times and a significantly smaller learning rate we were able to improve upon our data model. In the images you can see how the metal plates are properly labeled and the image with a lack of plate has no labeling. There are some false positives when the image is too blurry and there is an overwhelming presence of blue or red lights shown then the model will give it a false label. Overall, the data model performs excellent.
![https://media.discordapp.net/attachments/1095549530583343116/1100197031458377758/image.png?width=932&height=528](https://media.discordapp.net/attachments/1095549530583343116/1100197031458377758/image.png?width=932&height=528)

![https://media.discordapp.net/attachments/1095549530583343116/1100197186312089640/image.png?width=901&height=503](https://media.discordapp.net/attachments/1095549530583343116/1100197186312089640/image.png?width=901&height=503)

## Conclusion
In conclusion, we used a YOLO v5 model to successfully detect Robomaster armor plates on a Beaglebone AI-64. Our process consisted of labeling armor plates, using Open Neural Network Exchange to maximize the performance of the model, and training it on the Beaglebone. Aside from time constraints, the biggest difficulty was learning how to use YOLO and program it onto the Beaglebone. Another difficulty is the process of labeling images on CVAT. We believe we could improve upon our current YOLO v5 model by labeling more images but we were only given about 4000 images to work with from Purdue. In the end the latency for detection on YOLO was 7.56 ms which was right around the 7.5ms benchmark time. 

## Sources
The GitHub repo for the machine learning model YOLO v5 -  [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 

The BeagleBoard - [https://beagleboard.org/ai-64](https://beagleboard.org/ai-64) 

About YOLOv5 - [https://blog.roboflow.com/yolov5-improvements-and-evaluation/](https://blog.roboflow.com/yolov5-improvements-and-evaluation/) 

A guide for using YOLOv5 on the Beagleboard - [https://github.com/WasabiFan/tidl-yolov5-custom-model-demo](https://github.com/WasabiFan/tidl-yolov5-custom-model-demo)

Paul created two guides, one for [labeling images using CVAT](https://docs.google.com/document/d/1Dv-AmiVg_wX1D8T-aZfzkl14iyU11Dd2TfjmNSOxmIE/edit) and another for [installing the Beaglebone](https://docs.google.com/document/d/1eZiniOKkxNeE0nyb9P3TmpwKyLvC_PayqkE7lATLaPI/edit#heading=h.kjws9f15q4ae) 