import cv2
from yolo_predictions import YoloPredictions


save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz'
video_path = '/home/ellentuane/Documents/IC/videos/test.mp4'
labels_path = '/home/ellentuane/Documents/IC/coco.names'
cfg_path = '/home/ellentuane/Documents/IC/yolov4.cfg'
weight_path = '/home/ellentuane/Documents/IC/yolov4.weights'

def rescaleFrame(frame, scale=0.75):
    # images, video and live videos
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(960, 960), scale=1/255)

# Obter o nome das categorias
layer_names = YoloPredictions.layer_name(net)

classes = []
with open(labels_path, 'r') as file_object:
    for class_name in file_object:
        class_name = class_name.strip()
        classes.append(class_name)

print(classes)

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    frame = rescaleFrame(frame)

    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        name = classes[class_id]
        if name == 'person':
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 1)



    cv2.imshow('Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()