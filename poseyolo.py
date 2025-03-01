from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import cv2

class ypose:
    def __init__(self, verbose=True):
    # Load the YOLO model
        self.model = YOLO("models_1/best_y11.pt")
        self.model.to('cuda')
        self.verbose = verbose 

    def pose(self,pf, head_bbox):
        frame = pf
        results = self.model(pf)
        mydict = {}
        # Iterate over each detected person and print their keypoints
        for result in results:
            keypoints = result.keypoints  # Get keypoints as a numpy array or tensor
            if keypoints is not None and keypoints.data.shape[1]>0: 
                keypoints_data=keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    #Uncomment the following lines for key point visualization
                    for kp in person_keypoints:
                        x, y, confidence = kp
                        if confidence > 0.5:  # Optional: Only draw keypoints with sufficient confidence
                            # cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw the keypoint
                            cv2.drawMarker(frame, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)

                    # print("person", person_keypoints, "type", type(person_keypoints))
                    C = person_keypoints[0]  # Keypoint 0
                    A = person_keypoints[2]  # Keypoint 2
                    B = person_keypoints[3]  # Keypoint 3

                    Cx=int(C[0].item())
                    Cy=int(C[1].item())
                    Ax=A[0].item() 
                    Ay=A[1].item()
                    Bx=B[0].item()
                    By=B[1].item()

                    for key, value in head_bbox.items():
                        if (value[0] < Cx < value[2]) and (value[1]<Cy<value[3]):
                            id = key
#                    if head_bbox.get(1)[0] < Cx < head_bbox.get(1)[2]: id = 1
                            mydict[id] = {'Ax': Ax, 'Ay': Ay, 'Bx':Bx, 'By':By, 'Cx':Cx, 'Cy':Cy}
                            # print("Id assigned:",id,key)
                            break
                    
        return mydict
