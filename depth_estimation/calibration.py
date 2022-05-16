import cv2
import numpy as np
import os
import glob
# Define images path and matrix saving path
path = 'depth_estimation/images/'
save_path = 'depth_estimation/calibration_parameters'

# Define the dimensions of checkerboard
CHECKERBOARD = (9, 6)

# Set image type
images = glob.glob('*.HEIC') # set image type (.png, .jpg, .jpeg, .HEIC, ...)

#check if images were loaded
print(f"Total images loaded {images}")

# Find chess board corners
# stop the iteration when specified accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                      0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
count = 0

if os.path.exists(path):
    for img in images:
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
            cv2.imwrite(f"{save_path}/{img}.jpg", image)
            # print(filename)
            count += 1

        else:
            pass
            # If for any reason cv can't find the chess board corners on an especific image, it will return the image name
            print('error: ', img)

print(f'Chess board corners found in {count} images')

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

np.save('depth_estimation/calibration_parameters/ret.npy', ret)
np.save('depth_estimation/calibration_parameters/matrix.npy', matrix)
np.save('depth_estimation/calibration_parameters/distortion.npy', distortion)
np.save('depth_estimation/calibration_parameters/r_vecs.npy', r_vecs)
np.save('depth_estimation/calibration_parameters/t_vecs.npy', t_vecs)

# Re-projection error gives a good estimation of just how exact the found parameters are.
# The closer the re-projection error is to zero, the more accurate the parameters we found are.
mean_error = 0
for i in range(len(objectp3d)):
    imgpoints, _ = cv2.projectPoints(objectp3d[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(objectp3d[i], imgpoints, cv2.NORM_L2)/len(imgpoints)
    mean_error += error

total_error = mean_error/len(objectp3d)
print( "total error: {}".format(mean_error/len(objectp3d)))
np.save('depth_estimation/calibration_parameters/total_error.npy', total_error)
