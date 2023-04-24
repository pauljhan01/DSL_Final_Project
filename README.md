# DSL_Final_Project
# Labeling Armor Plates using YOLO v5 and Training that Model on the Beaglebone AI-64
By Paul Han, Drew Conyers, Michael Routh, Tristan Becnel, Kenneth Pinzon, Isabel Aguilera, Tanya Shiramagond, and Anthony Lam

![https://cdn.discordapp.com/attachments/1095549530583343116/1100186102838607982/Fleet_Render_1x1.png](https://cdn.discordapp.com/attachments/1095549530583343116/1100186102838607982/Fleet_Render_1x1.png)

## Section 1: Problem Overview
In this project, we used an object detection machine learning algorithm called YOLO v5 (You Only Look Once) for detecting Robomaster armor plates with high, consistent accuracy. We then enhanced its performance on a System-on-Chip (SoC) called Beaglebone AI-64, equipped with an Intel Realsense camera, to achieve very high frames per second. Our approach involves using ONNX (Open Neural Network Exchange) files, which accelerate matrix operations and inference for faster calculations. To accomplish this, we collected and labeled images of armor places attached to competition robots and trained YOLO v5 using these images in a supervised learning manner. Our dataset comprises images provided by Purdue University’s Boilderbots and the University of Texas’ Stampede Robomaster teams, which were labeled using Computer Vision Annotation Tool (CVAT). We used approximately four thousand images for training. The YOLO v5 model, pre-trained on the COCO dataset, is retrained on the Robomaster dataset to detect armor places, a critical component of Robomaster competition, as damage in the game is only incurred when armor places are hit. Using YOLO API, we modified and tailored the YOLO v5 model to suit our project’s requirements.

## Section 2: YOLO v5 Overview
YOLO v5 is a computer vision model and is used for object detection. There is a small, medium, large, and extra-large version, where accuracy increases with size. YOLOv5 performs well relative to its inference time, so it is known to process images quickly with a high degree of accuracy.
The YOLO v5 network is made of 3 pieces: the backbone, the neck, and the head. The backbone is a convolutional neural network that takes the image input and forms image features by putting it through different layers. The neck merges these image features together to make a prediction. The head forms a class prediction based on the acquired features, which overall creates a sparse prediction of the location of the detected object.
YOLO v5 uses data augmentation and loss calculation techniques as their main training procedures. In our case, the YOLO model took the Robomaster images with labeled armor plates and used these training procedures to learn how to detect the armor plates. 


## Section 3: Data Collection and Preprocessing
We received roughly 4000 training images from Purdue with the agreement that we label them and send them the labeled images. The images that were provided consisted of metal plates that had blue lights and of metal plates that had red lights. We then proceeded to label these images using a publicly available tool called CVAT (Computer Vision Annotation Tool, cvat.ai). We created two distinct labels to distinguish these plates with a red label for red lights and a blue label for blue lights. Each team member was responsible for labeling ~500 images. Using CVAT, we each labeled 500 plates by drawing squares around the metal plates. A blue label if the plate had blue lights and a red label if the plate had red lights. If the image was too blurry or both lights on the plate were not visible then we would skip that image. Once we were finished labeling the images we exported the data as YOLO 1.1. Once every team member was finished we were now ready to start training and optimizing our model.

## Section 4: Performance Optimization with ONNX
ONNX or Open Neural Network Exchange is an open-source library dedicated to facilitating the exchange of neural networks between different frameworks and tools. The library allows for the expression of different deep learning models to be represented in a common format which allows models to be used across platforms and multiple frameworks. 

Our team is capitalizing on ONNX’s efficiency and model optimization to maximize our YOLO's performance. ONNX supports hardware acceleration on specialized accelerators which boosts stellar performance due to the standardization of model formatting. ONNX also supports quantization, reducing memory usage and computation time as well as having the ability of ‘pruning’ which gets rid of  unnecessary model complexity.

Lastly, although there may be other machine learning model optimization libraries like  NVIDIA’s CUDA, our model was made by Texas Instruments and our beaglebone has proprietary TI chips integrated within the SoC. Therefore the decision to use ONNX is obvious. The library is the most compatible with the model, hardware, and can push the most performance out of our project. 


## Section 5: Integrating with Beaglebone AI-64 and Intel Realsense Camera

## Section 6: Test and Evaluation

## Conclusion
