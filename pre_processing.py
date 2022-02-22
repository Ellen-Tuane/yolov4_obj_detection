import cv2


class PreProcess:

    def __init__(self):
        pass

    @staticmethod
    def blob_net(image, layer_names, net, net_height, net_width):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (net_height, net_width), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(layer_names)

        return outputs

    @staticmethod
    def remove_shade(image):
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        mask = object_detector.apply(image)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        return mask


