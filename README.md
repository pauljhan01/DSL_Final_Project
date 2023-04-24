# DSL_Final_Project
# Labeling Armor Plates using YOLO v5 and Training that Model on the Beaglebone AI-64
By Paul Han, Drew Conyers, Michael Routh, Tristan Becnel, Kenneth Pinzon 

https://cdn.discordapp.com/attachments/1095549530583343116/1100186102838607982/Fleet_Render_1x1.png 

## Section 1: Problem Overview
In this project, we used an object detection machine learning algorithm called YOLO v5 (You Only Look Once) for detecting Robomaster armor plates with high, consistent accuracy. We then enhanced its performance on a System-on-Chip (SoC) called Beaglebone AI-64, equipped with an Intel Realsense camera, to achieve very high frames per second. Our approach involves using ONNX (Open Neural Network Exchange) files, which accelerate matrix operations and inference for faster calculations. To accomplish this, we collected and labeled images of armor places attached to competition robots and trained YOLO v5 using these images in a supervised learning manner. Our dataset comprises images provided by Purdue University’s Boilderbots and the University of Texas’ Stampede Robomaster teams, which were labeled using Computer Vision Annotation Tool (CVAT). We used approximately four thousand images for training. The YOLO v5 model, pre-trained on the COCO dataset, is retrained on the Robomaster dataset to detect armor places, a critical component of Robomaster competition, as damage in the game is only incurred when armor places are hit. Using YOLO API, we modified and tailored the YOLO v5 model to suit our project’s requirements.

## Section 2: YOLO v5 Overview
YOLO v5 is a computer vision model and is used for object detection. There is a small, medium, large, and extra-large version, where accuracy increases with size. YOLOv5 performs well relative to its inference time, so it is known to process images quickly with a high degree of accuracy.

## Section 3: Data Collection and Preprocessing
We received roughly 4000 training images from Purdue with the agreement that we label them and send them the labeled images. The images that were provided consisted of metal plates that had blue lights and of metal plates that had red lights. We then proceeded to label these images using a publicly available tool called CVAT (Computer Vision Annotation Tool, cvat.ai). We created two distinct labels to distinguish these plates with a red label for red lights and a blue label for blue lights. Each team member was responsible for labeling ~500 images. Using CVAT, we each labeled 500 plates by drawing squares around the metal plates. A blue label if the plate had blue lights and a red label if the plate had red lights. If the image was too blurry or both lights on the plate were not visible then we would skip that image. Once we were finished labeling the images we exported the data as YOLO 1.1. Once every team member was finished we were now ready to start training and optimizing our model.

## Section 4: Performance Optimization with ONNX

## Section 5: Integrating with Beaglebone AI-64 and Intel Realsense Camera

## Section 6: Test and Evaluation

## Conclusion
