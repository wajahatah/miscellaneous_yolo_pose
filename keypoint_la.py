from ultralytics import YOLO
import os
import cv2
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # Load your trained YOLOv8 model
    # model = YOLO("runs/pose/trailv11-3/weights/best11_v3.pt")
    model = YOLO("bestv6-1.pt")

    video_path = "D:/Wajahat/la_chunks/test_bench_3/chunk_24-02-25_11-41.avi"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available
        frame = cv2.resize(frame,(1280,720))
        results = model(frame)
        for result in results:
            keypoints = result.keypoints  # Access the keypoints object
            if keypoints is not None:
                keypoints_data = keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    keypoint_list = []
                    for kp_idx, keypoint in enumerate(person_keypoints):
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255,0), -1)
                        keypoint_list.append((x,y,confidence))

        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()