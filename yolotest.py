"""from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

'''def find_perpendicular_point(A, B, C):
    ""
    Finds the point on line segment AB where a perpendicular from point C would intersect.
    
    Parameters:
    A (tuple): Coordinates of point A (x_A, y_A).
    B (tuple): Coordinates of point B (x_B, y_B).
    C (tuple): Coordinates of point C (x_C, y_C).
    
    Returns:
    tuple: Coordinates of the perpendicular intersection point on AB.
    ""
    # Unpack the coordinates
    A[0], A[1] = A
    x_B, y_B = B
    x_C, y_C = C
    
    # Calculate the direction vector from A to B
    AB_x = x_B - x_A
    AB_y = y_B - y_A
    
    # Calculate the vector from A to C
    AC_x = x_C - x_A
    AC_y = y_C - y_A
    
    # Projection formula to find the scalar t for point D on AB
    t = (AC_x * AB_x + AC_y * AB_y) / (AB_x ** 2 + AB_y ** 2)
    
    # Calculate the coordinates of the perpendicular point D
    x_D = x_A + t * AB_x
    y_D = y_A + t * AB_y
    
    return (x_D, y_D)
'''

def visual_region(center_point, angle):
    length = 150  # Length of the lines

    # Convert degrees to radians
    angle_rad = angle*(3.142/180)

    # Calculate the endpoint for the 30-degree line
    left_end_point_1 = (
        int(center_point[0] - (length * math.sin(angle_rad))),
        int(center_point[1] - length * math.cos(angle_rad))
    )

    # Calculate the endpoint for the -30-degree line
    right_end_point_2 = (
        int(center_point[0] + (length * math.sin(angle_rad))),
        int(center_point[1] - (length * math.cos(angle_rad)))
    )

    return left_end_point_1, right_end_point_2


if __name__ == "__main__":

    # model = YOLO("runs/pose/train25/weights/best.pt")
    model = YOLO("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/runs/pose/trail4/weights/best_y11.pt")

    # results = model("Cam_19_14.mp4", save = True, show = True, name = 'cam14')
    # model = YOLO("C:/Users/LAMBDA THETA/Downloads/ultralytics-main/ultralytics-main/modelsv8/yolov8l-pose.pt")
    # model.to('cuda')
    
    # model.train(data="C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/batch_5/config.yaml", epochs = 100, batch=16, patience=20, imgsz = 640, device=0, box=0.01, dfl=0.01, warmup_epochs= 1)
    # model.train(data="C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/batch_5(0)/batch_5/config.yaml", epochs = 1, patience=20, imgsz = 640) #, device=0, box=0.01, dfl=0.01, warmup_epochs= 1)
    
    
    # print("result:" , results[0])
    video_path = "Cam_19_14.mp4"
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter('output_with_keypoints.mp4', fourcc, fps, (frame_width, frame_height))

    # Open a text file to save keypoints
    with open('keypoints_output2.txt', 'w') as f:

        # Check if the video capture opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_count = 0  # To track frame number

    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Run inference on the current frame
        results = model(frame)

        # Iterate over each detected person and print their keypoints
        for result in results:
            keypoints = result.keypoints  # Get keypoints as a numpy array or tensor
            # print("Keypoints structure:", keypoints)

            if keypoints is not None: # and isinstance(keypoints, np.ndarray):
                keypoints_data=keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    f.write(f"Frame {frame_count}, Person {person_idx}:/n")
                    for kp_idx, keypoint in enumerate(person_keypoints):
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()

                        # Save the keypoints to the text file
                        f.write(f"  Keypoint {kp_idx}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})\n")

                        # Draw a circle at each keypoint
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                        # Put the keypoint values on the frame
                        cv2.putText(
                            frame,
                            f"({int(x)}, {int(y)}, {confidence:.2f})",
                            (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 0, 0),
                            1
                        )

            out.write(frame)                            
            
            # for result in results:
            #     annotated_frame = result.plot()  # Draw keypoints and bounding boxes on the frame
            cv2.imshow('Pose Detection', frame)

            # Press 'q' to quit the video display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    # Release video capture and close display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    ""
                    # C = person_keypoints[0]  # Keypoint 0
                    # A = person_keypoints[2]  # Keypoint 2
                    # B = person_keypoints[3]  # Keypoint 3

                    # Cx=C[0].item()
                    # Cy=C[1].item()
                    # Ax=A[0].item() 
                    # Ay=A[1].item()
                    # Bx=B[0].item()
                    # By=B[1].item()

                    # AB_x = Bx - Ax
                    # AB_y = By - Ay
                    
                    # # Calculate the vector from A to C
                    # AC_x = Cx - Ax
                    # AC_y = Cy - Ay
                    
                    # # Projection formula to find the scalar t for point D on AB
                    # # if Cx or Cy or Ax or Ay or Bx or By is not None:
                    # if all(value != 0 for value in [Cx, Cy, Ax, Ay, Bx, By]):
                    #     t = (AC_x * AB_x + AC_y * AB_y) / (AB_x ** 2 + AB_y ** 2)
                    
                    # # Calculate the coordinates of the perpendicular point D
                    # x_D = Ax + t * AB_x
                    # y_D = Ay + t * AB_y

                    # print("D coordinates:", x_D,y_D)
                    # cv2.circle(frame, (int(x_D), int(y_D)), 5, (254, 32, 32), -1)  # Keypoint D

                    # cv2.line(frame, (int(Ax), int(Ay)), (int(Bx), int(By)), (0, 255, 0),2)  # Green line
                
                    # # Draw line from C to D
                    # cv2.line(frame, (int(Cx), int(Cy)), (int(x_D) ,int(y_D)), (255, 0, 0),2)  # Blue line

                    # center = int(x_D) , int(y_D)
                    # LA_angle_threshold = 70
                    # frame_visual_area = frame

                    # left_point, right_point = visual_region(center, LA_angle_threshold)
                    # frame_visual_area = cv2.line(frame_visual_area, center, left_point, (255, 255, 255), 4)
                    # frame_visual_area = cv2.line(frame_visual_area, center, right_point, (255, 255, 255), 4)

                        # Save the keypoints (x, y) values to the text file
                        # f.write(f"Frame {frame_count}, Person {person_idx}:\n")
                        # f.write(f"  Keypoint 0: (x={C[0].item():.2f}, y={C[1].item():.2f})\n")
                        # f.write(f"  Keypoint 2: (x={A[0].item():.2f}, y={A[1].item():.2f})\n")
                        # f.write(f"  Keypoint 3: (x={B[0].item():.2f}, y={B[1].item():.2f})\n")

                        # # Draw circles at the selected keypoints
                        # cv2.circle(frame, (int(C[0]), int(C[1])), 5, (0, 255, 0), -1)  # Keypoint 0
                        # cv2.circle(frame, (int(A[0]), int(A[1])), 5, (0, 0, 255), -1)  # Keypoint 2
                        # cv2.circle(frame, (int(B[0]), int(B[1])), 5, (255, 0, 0), -1)  # Keypoint 3

                        # # Put the keypoint values on the frame
                        # cv2.putText(frame, f"K0({int(C[0])}, {int(C[1])})", (int(C[0]) + 5, int(C[1]) - 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        # cv2.putText(frame, f"K2({int(A[0])}, {int(A[1])})", (int(A[0]) + 5, int(A[1]) - 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        # cv2.putText(frame, f"K3({int(B[0])}, {int(B[1])})", (int(B[0]) + 5, int(B[1]) - 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # find_perpendicular_point(A, B, C)

                        
                        print(f"Person {person_idx + 1}:")
                        for kp_idx, keypoint in enumerate(person_keypoints):
                            x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                            print(f"  Keypoint {kp_idx + 1}: x={x}, y={y}, confidence={confidence:.2f}")

                            f.write(f"  Keypoint {kp_idx}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})\n")

                            
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                            # Put the keypoint values on the frame
                            cv2.putText(
                                frame,
                                f"({int(x)}, {int(y)}, {confidence:.2f})",
                                (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 0),
                                1
                            )
                # else:
                #     print("No keypoints detected")
                    # for i, keypoint in enumerate(keypoints):
                    #     # Ensure the keypoint has at least 2 values (x, y)
                    #     if len(keypoint) >= 2:
                    #         x, y = keypoint[0], keypoint[1]  # Extract x and y coordinates
                    #         print(f"Keypoint {i}: x={x}, y={y}")
                    #     else:
                    #         print(f"Keypoint {i} has insufficient data")

            # Display the frame with keypoints (optional)
            out.write(frame)                            
            ""

'''
from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":

    model = YOLO('yolov8l-pose.pt')

    model.train(data="C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/batch_5/config.yaml", freeze = 6, epochs = 200,patience = 20,  imgsz = 640)
    '''
"""

from ultralytics import YOLO
import os
import cv2
import numpy as np

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_lines(frame, keypoints, connections):
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]

            # Ensure confidence is above a threshold to draw
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Yellow lines

if __name__ == "__main__":
    # Load your trained YOLOv8 model
    model = YOLO("runs/pose/trailv11-3/weights/best11_v3.pt")

    # Open the video file
    # video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_19_02.mp4"#"Cam_19_14.mp4"
    video_path = "C:/Users/LAMBDA THETA/Videos/batch7-RTSP112/batch_7_13.mp4"#"Cam_19_14.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video properties for saving the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter('output_with_keypoints.mp4', fourcc, fps, (frame_width, frame_height))

    connections = [
        (0, 1), (0, 2), (0, 3),  # Keypoint 0 to 1, 2, and 3
        (3, 4), (3, 5),          # Keypoint 1 to 4 and 7
        (5, 6), (6, 9),           # Keypoint 4 to 5, and 5 to 6
        (7, 8), (4, 7)           # Keypoint 7 to 8, and 8 to 9
    ]

    # Open a text file to save keypoints
    with open('keypoints_output.txt', 'w') as f:

        # Check if the video capture opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_count = 0  # To track frame number

    # Read and process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available
            
            black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Run inference on the current frame
            frame = cv2.resize(frame,(1280,720))
            results = model(frame)

            # Iterate over each detected object and print their keypoints
            for result in results:
                keypoints = result.keypoints  # Access the keypoints object

                if keypoints is not None:
                    # Get the data attribute, which contains x, y, and confidence values
                    keypoints_data = keypoints.data
                    for person_idx, person_keypoints in enumerate(keypoints_data):
                        f.write(f"Frame {frame_count}, Person {person_idx}:\n")
                        keypoint_list = []

                        for kp_idx, keypoint in enumerate(person_keypoints):
                            x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                            keypoint_list.append((x,y,confidence))

                            # Save the keypoints to the text file
                            f.write(f"  Keypoint {kp_idx}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})\n")

                            # Draw a circle at each keypoint
                            # cv2.circle(black_frame, (int(x), int(y)), 5, (0,0, 255), -1)

                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255,0), -1)
                            # draw_lines(frame,keypoint_list,connections)

                            # Put the keypoint values on the frame
                            # cv2.putText(
                            #     frame,
                            #     f"({int(x)}, {int(y)}, {confidence:.2f})",
                            #     (int(x) + 5, int(y) - 5),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.4,
                            #     (255, 0, 0),
                            #     1
                            # )

                            # if confidence > 0.5:
                            #     points.append((int(x),int(y)))

                            # else:
                            #     points.append(None)

                            # for i in range (len(points) -1):
                            #     if points[i] and points[i+1]:
                            #         cv2.line(frame, points[i], points[i+1], (0,225,0),2)

        # Write the processed frame to the output video file
            out.write(frame)
            frame_filename = os.path.join("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/f1-v1", f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename,frame)
            out.write(frame)

            # Display the frame with keypoints and values
            cv2.imshow('Pose Detection', frame)
            # cv2.imshow('Pose Detection', black_frame)

            # Delay of 50 milliseconds to slow down the video playback
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            frame_count += 1  # Increment frame counter

    # Release the video capture and close display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
