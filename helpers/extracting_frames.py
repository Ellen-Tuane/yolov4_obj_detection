from frame import Frame

save_path = '/home/ellentuane/Documents/IC/yolov4_obj_detection/helpers/output'
video_path = '//home/ellentuane/Documents/IC/videos/stereo_left_camera/Left_0.MOV'

Frame.video_frame(video_path, save_path, 30)
print('Done!')

