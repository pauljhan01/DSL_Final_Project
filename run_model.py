## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2.pyrealsense2 as rs
import onnxruntime as rt
import numpy as np
import cv2

CONFIDENCE_THRESHOLD = 0.3

def render_boxes(image, inference_width, inference_height, output):
	assert len(output.shape) == 3
	output_count = output.shape[1]

	for i in range(output_count):
		x1, y1, x2, y2, confidence, class_idx_float = output[0, i, :]
		if confidence <= CONFIDENCE_THRESHOLD:
			continue

		x1 = int(round(x1 / inference_width * image.shape[1]))
		y1 = int(round(y1 / inference_height * image.shape[0]))
		x2 = int(round(x2 / inference_width * image.shape[1]))
		y2 = int(round(y2 / inference_height * image.shape[0]))

		# Yes, TI outputs the class index as a float...
		class_draw_color = {
			# Colors for boxes of each class, in (R, G, B) order.
			0.: (255, 50, 50),
			1.: (50, 50, 255),
			# TODO: if using more than two classes, pick some more colors...
		}[class_idx_float]

		# Reverse RGB tuples since OpenCV images default to BGR
		cv2.rectangle(image, (x1, y1), (x2, y2), class_draw_color[::-1], 3)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
	if s.get_info(rs.camera_info.name) == 'RGB Camera':
		found_rgb = True
		break
if not found_rgb:
	print("The demo requires Depth camera with Color sensor")
	exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
	config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

model_path = "tidl-yolov5-custom-model-demo/docker/artifacts/last_with_shapes.onnx"
artifacts_dir = "tidl-yolov5-custom-model-demo/docker/artifacts/tidl_output"
so = rt.SessionOptions()

print("Available execution providers : ", rt.get_available_providers())

runtime_options = {
	"platform": "J7",
	"version": "8.2",

	"artifacts_folder": artifacts_dir,
	#"enableLayerPerfTraces": True,
}

desired_eps = ['TIDLExecutionProvider','CPUExecutionProvider']
sess = rt.InferenceSession(
	model_path,
	providers=desired_eps,
	provider_options=[runtime_options, {}],
	sess_options=so
)

input_details, = sess.get_inputs()
batch_size, channel, height, width = input_details.shape
print(f"Input shape: {input_details.shape}")

assert isinstance(batch_size, str) or batch_size == 1
assert channel == 3

input_name = input_details.name
input_type = input_details.type

print(f'Input "{input_name}": {input_type}')

try:
	while True:

		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not color_frame:
			continue

		# Convert images to numpy arrays

		color_image = np.asanyarray(color_frame.get_data())

		input_data = cv2.resize(color_image, (width, height)).transpose((2, 0, 1))[::-1, :, :] / 255
		input_data = input_data.astype(np.float32)
		input_data = np.expand_dims(input_data, 0)

		print("I make it here")

		detections, = sess.run(None, {input_name: input_data})

		print("After I made it there, I made it here")
		render_boxes(color_image, width, height, detections[0, :, :, :])
		# Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', color_image)
		cv2.waitKey(1)

finally:

	# Stop streaming
	pipeline.stop()