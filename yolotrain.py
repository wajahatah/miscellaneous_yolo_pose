from ultralytics import YOLO
# from ultralytics.data.dataset import LoadImagesAndLabels
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# class SkipFramesDataset(LoadImagesAndLabels):
#     def __init__(self, *args, skip_rate=5, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Keep only frames that are NOT multiples of skip_rate
#         self.img_files = [img for idx, img in enumerate(self.img_files) if (idx % skip_rate) != 0]
#         self.label_files = [lbl for idx, lbl in enumerate(self.label_files) if (idx % skip_rate) != 0]

if __name__ == "__main__":

    # model = YOLO("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/weights/best.pt")
    model = YOLO("yolo11l-pose.pt")
    model.train(data="/home/haithem-ge/wajahat/pose/data.yaml", epochs = 300, 
                batch = 0.95, imgsz = 640, patience=5, lr0 = 0.001, 
                name = "7thiteration", device =0, workers=8)
                # dataset=SkipFramesDataset, dataset_kwargs={"skip_rate": 5})

    # C:\OsamaEjaz\Qiyas_Gaze_Estimation\miscellaneous_yolo_pose\updated_dataset\data.yaml 