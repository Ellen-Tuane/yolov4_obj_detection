import os
import cv2


class Frame:
    def __init__(self):
        pass

    @staticmethod
    def create_dir(path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            print(f"ERROR: creating directory with name {path}")

    @staticmethod
    def save_frame(video_path, save_dir, gap, frame, idx):
        name = video_path.split("/")[-1].split(".")[0]
        save_path = os.path.join(save_dir, name)
        Frame.create_dir(save_path)
        if idx == 0:
            cv2.imwrite(f"{save_path}/{name}_{idx}_.jpg", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{name}_{idx}_.jpg", frame)

    @staticmethod
    def video_frame(video_path, save_dir, gap):
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if ret:
                Frame.save_frame(video_path, save_dir, gap, frame, idx)
            else:
                cap.release()
                break
            idx += 1

        return False
