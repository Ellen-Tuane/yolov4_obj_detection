from frame import Frame

save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz/'
video_path = '/home/ellentuane/Documents/IC/videos/Aerial_City.mp4'
ver = True

while ver:
    ver = Frame.video_frame(video_path, save_path, 30)

