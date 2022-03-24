import os
import cv2


class Frame:
    def __init__(self):
        pass

    @staticmethod
    def create_dir(input_path, output_path):
        name = input_path.split("/")[-1].split(".")[0]
        save_path = os.path.join(output_path, name)
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            print(f"ERROR: creating directory with name {save_path}")
        return save_path, name

    @staticmethod
    def save_frame(save_path, name, gap, frame, idx):
        if idx == 0:
            cv2.imwrite(f"{save_path}/{name}_{idx}_.jpg", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{name}_{idx}_.jpg", frame)

    @staticmethod
    def video_frame(video_path, save_dir, gap):
        cap = cv2.VideoCapture(video_path)
        idx = 0
        save_path, name = Frame.create_dir(video_path, save_dir)
        while True:
            ret, frame = cap.read()
            if ret:
                Frame.save_frame(save_path, name, gap, frame, idx)
            else:
                cap.release()
                break
            idx += 1

    @staticmethod
    def rescaleFrame(frame, scale=0.75):
        # images, video and live videos
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

