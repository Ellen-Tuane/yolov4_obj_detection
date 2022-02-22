import cv2



save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz'
video_path = '/home/ellentuane/Documents/IC/videos/test.mp4'
labels_path = '/home/ellentuane/Documents/IC/coco.names'
cfg_path = '/home/ellentuane/Documents/IC/yolov4.cfg'
weight_path = '/home/ellentuane/Documents/IC/yolov4.weights'

def rescaleFrame(frame, scale=0.5):
    # images, video and live videos
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Object detector from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

cap = cv2.VideoCapture(video_path)

stop = 0
i = 0
while True:
    if stop == 0:
        ret, frame = cap.read()

        # Resizing
        frame = rescaleFrame(frame)

        # Extract region of interesting and then apply it to mask
        #roi = frame[]



        # Object detection
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        '''contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 75:
                #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)'''






        cv2.imshow('frame', frame)
        cv2.imshow('Mask', mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            stop = not stop
        if key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()