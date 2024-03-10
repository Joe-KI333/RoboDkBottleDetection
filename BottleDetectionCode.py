from robodk.robolink import *  # RoboDK API
from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
from robolink import import_install

#----------------------------------
# Settings
PROCESS_COUNT = -1  # How many frames to process before exiting. -1 means indefinitely.
OUTPUT_DIR = "output_images"  # Directory to save the inference results

CAM_NAME = "camera"  # Change the camera name here to match your camera name

#DISPLAY_SETTINGS = True
WDW_NAME_PARAMS = 'RoboDK - Blob detection parameters'

DISPLAY_RESULT = True
WDW_NAME_RESULTS = 'RoboDK - Blob detections'

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#----------------------------------
# Get the simulated camera from RoboDK
RDK = Robolink()

cam_item = RDK.Item(CAM_NAME, ITEM_TYPE_CAMERA)
if not cam_item.Valid():
    raise Exception("Camera not found! %s" % CAM_NAME)
cam_item.setParam('Open', 1)  # Force the camera view to open

# Load a pretrained YOLOv5 model
model = YOLO('C:/Users/joelb/OneDrive/Documents/RoboDK/bottle.pt')
classnames = ['label','no_label']

#----------------------------------------------
# Set up the detector with default parameters.
# Default parameters extract dark circular blobs.
params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector_create(params)

#----------------------------------------------
# Start processing camera frames
count = 0
while count < PROCESS_COUNT or PROCESS_COUNT < 0:
    print("=============================================")
    print("Processing image %i" % count)
    count += 1

    # Get the image from RoboDK
    bytes_img = RDK.Cam2D_Snapshot("", cam_item)
    if bytes_img == b'':
        raise Exception("Failed to capture image from camera.")
    nparr = np.frombuffer(bytes_img, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Ensure image has 3 channels (RGB)
    if frame is None or frame.shape[2] != 3:
        raise Exception("Failed to decode image or incorrect number of channels.")

    # Perform blob detection
    keypoints = detector.detect(frame)

    # Perform object detection using YOLOv5 model
    results = model(frame)  # list of Results objects

    # Draw bounding boxes around the detected objects
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_detect}', (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_DUPLEX, 1, (255,165,0), 1, cv2.LINE_AA)

    # Save the inference result
    output_path = os.path.join(OUTPUT_DIR, f"C:/RoboDK/inference/result_{count}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Inference result saved to: {output_path}")

    # Display the detection results
    if DISPLAY_RESULT:
        # Resize the image
        frame = cv2.resize(frame, (int(frame.shape[1] * .75), int(frame.shape[0] * .75)))

        cv2.imshow(WDW_NAME_RESULTS, frame)
        key = cv2.waitKey(500)
        if key == 27:
            break  # User pressed ESC, exit
        if cv2.getWindowProperty(WDW_NAME_RESULTS, cv2.WND_PROP_VISIBLE) < 1:
            break  # User closed the window, exit

cv2.destroyAllWindows()
