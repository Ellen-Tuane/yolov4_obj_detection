
def find_depth(right_point, left_point, baseline, fl_pixel):

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    # If you substitute your units you get:
    # depth = baseline (meter) * focal length (pixel) / disparity-value (pixel).
    # So the result is in meters because pixels are canceled down.
    z_Depth = (baseline * fl_pixel)/disparity             #Depth in [cm]

    return z_Depth