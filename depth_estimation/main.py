import cv2
import undistort
import triangulation
import numpy as np
from Detections.yolo_predictions import YoloPredictions

# Camera parameters path
matrix_path ='depth_estimation/calibration_parameters/matrix.npy'
dist_path = 'depth_estimation/calibration_parameters/distortion.npy'

# setting images path
save_path = 'depth_estimation/output'
image_right = 'helpers/output/right_5cm/right_5cm_570_.jpg'
image_left ='helpers/output/Left_0/Left_0_900_.jpg'

# YOLO parameters
classes_path = 'Detections/classes/coco.names'
cfg_path = 'Detections/cfg/yolov4-tiny.cfg'
weight_path = 'Detections/weights/yolov4-tiny.weights'
labels = open(classes_path).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
layer_names = YoloPredictions.layer_name(net)

# Stereo vision parameters
B = 5  # Distance between the cameras [cm]
fl = 3104.4  # right camera focal length [px]

# Open both cameras
cap_right = cv2.imread(image_right)
cap_left = cv2.imread(image_left)

# Initial values
count = -1
while True:
    count += 1

    # Undistortion
    frame_right = undistort.undistorted(cap_right, matrix_path, dist_path)
    frame_left = undistort.undistorted(cap_right, matrix_path, dist_path)

    # If cannot catch any frame, break
    if cap_right == False or cap_left == False:
        break
    else:
        # APPLY YOLO DETECTION AND RETURN BBOX CENTER



        ################## CALCULATING BALL DEPTH #########################################################

        # If no ball can be caught in one camera show text "TRACKING LOST"
        if np.all(circles_right) == None or np.all(circles_left) == None:
            cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.

            depth = triangulation.find_depth(circles_right, circles_left, frame_right, frame_left, B, f)

            cv2.putText(frame_right, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(frame_left, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(frame_right, "Distance: " + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (124, 252, 0), 2)
            cv2.putText(frame_left, "Distance: " + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (124, 252, 0), 2)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("Depth: ", depth)

        # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)
        cv2.imshow("mask right", mask_right)
        cv2.imshow("mask left", mask_left)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
'''