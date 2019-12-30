# import the necessary packages
from modules.centroidtracker import CentroidTracker
from modules.trackableobject import TrackableObject
import numpy as np
import argparse
import time
import dlib
from modules import utils

from cv2 import cv2

# Create argument parser and parse argumemts
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-s", "--skip_frames", type=int, default=20, help="# of skip frames between detections")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")

args = vars(ap.parse_args())

# Path to config file and weights trained on COCO dataset
config_path = "config/yolov3.cfg"
weigth_path = "config/yolov3.weights"
LABELS = open("config/coco.names").read().strip().split("\n")

# initialize a list of colors to represent each class label
np.random.seed(9)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# Load darknet model using dnn with specified weight
print("[INFO] loading YOLO from disk...")
darknet = cv2.dnn.readNetFromDarknet(config_path, weigth_path)

# Determine only the output layer names that we need from YOLO
ln = darknet.getLayerNames()
ln = [ln[i[0] - 1] for i in darknet.getUnconnectedOutLayers()]

# initialize the video stream of input video
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of cars that have crossed the counting line
totalFramesProcessed = 0
car_count = 0
# try to determine the total number of frames in the video file
total = utils.get_total_frames(vs)

while True:
    # Read frame from file
    (grabbed, frame) = vs.read()

    # if frame is not grabbed then break
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    rects = []
    rects_meta = []
    elap = 0

    if totalFramesProcessed % args["skip_frames"] == 0:
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Frame", frame)
        # cv2.waitKey(1)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        darknet.setInput(blob)
        start = time.time()
        layers_output = darknet.forward(ln)
        end = time.time()

        elap = (end - start)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layers_output:
            # loop over each of the detections
            for detection in output:

                # Extract confidence and classID from detections object
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # classID += 2
                # print("------", LABELS[classID])

                if LABELS[classID] != "car":
                    continue

                # Filter out week predictions
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[:4] * np.array([W, H, W, H])
                    (cX, cY, width, height) = box.astype("int")

                    # Get Top Left coordinates
                    x = int(cX - (width / 2))
                    y = int(cY - (height / 2))

                    # update our list of Boxes, confidences and classIDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x+w, y+h)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                tracker_meta = {
                    'tracker': tracker,
                    'start': (x, y),
                    'end': (x+w, y+h),
                    'color': [int(c) for c in COLORS[classIDs[i]]],
                    'label': LABELS[classIDs[i]],
                    'confidence': confidences[i]
                }

                trackers.append(tracker_meta)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker_meta in trackers:

            # update the tracker and grab the updated position
            tracker = tracker_meta['tracker']
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
            rects_meta.append(tracker_meta)

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'

    cv2.line(frame, (0, H - 50), (W, H - 50), (0, 255, 255), 2)

    if len(rects) == 0:
        print("[WARNING] bounding boxes are empty for frame - {}".format(totalFramesProcessed))
        continue

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    (objects, objectMeta) = ct.update(rects, rects_meta)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                # if direction < 0 and centroid[1] < H // 2:
                #     totalUp += 1
                #     to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                if direction > 0 and centroid[1] > H // 2:
                    car_count += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # draw a bounding box rectangle and label on the image
        meta = objectMeta[objectID]
        color = meta['color']
        (x,y) = meta['start']
        cv2.rectangle(frame, meta['start'], meta['end'], color, 2)
        text = "{}: {:.4f}".format(meta['label'], meta['confidence'])
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, "Total Count: {}".format(car_count), (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * (total/args["skip_frames"])))

    # write the output frame to disk
    writer.write(frame)
    totalFramesProcessed += 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()


