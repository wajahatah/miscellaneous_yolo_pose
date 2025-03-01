from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":

    # model = YOLO("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/weights/best.pt")
    model = YOLO("yolo11l.pt")
    # model.summary()
    print(model)


    # model.train(data="C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/updated_dataset/data.yaml", epochs = 300, imgsz = 640, freeze =6, patience=45, lr0 = 0.0001, name = "trailv8", device =0)