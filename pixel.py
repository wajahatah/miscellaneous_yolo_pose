import cv2
import os

# Callback function to capture mouse click events
def click_event(event, x, y, flags, param):
 
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the frame from the parameters
        frame = param
        # Get the pixel value at the clicked position (BGR format)
        pixel_value = frame[y, x]
        print(f"Clicked at (x={x}, y={y}) - Pixel value: {pixel_value}")

        # Optionally display the click position and pixel value on the frame
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Mark the click with a green circle
        text = f"x:{x},y:{y}"
        cv2.putText(frame, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Frame', frame)

# Load the video
input_dir = "C:/Users/LAMBDA THETA/Downloads/rtsp_images"
image_name = "101.jpg"
image_path = os.path.join(input_dir, image_name)
image = cv2.imread(image_path)

#to get the pixel values from the video frame uncomment the following lines
# video logic starts here 
# video_path = "E:/Wajahat/la_chunks/rtsp_61/chunk_27-01-25_13-40.avi"
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error: Could not open video.")
# else:
#     # Read the first frame
#     ret, frame = cap.read()
#     if ret:
#         # Display the frame
#         cv2.imshow('Frame', frame)
#         print("Click on the frame to get pixel values and coordinates.")

# end of video logic

#image logic starts here
if image is not None:
        
    cv2.imshow('Frame', image)
    print("Click on the image to get pixel values and coordinates.")

    cv2.setMouseCallback('Frame', click_event, image)

    cv2.waitKey(0)
# logic ends here

#uncomment the following lines to get the pixel values from the video frame
        # Set the mouse callback
#         cv2.setMouseCallback('Frame', click_event, image)
        
#         # Wait for a key press or mouse interaction
#         cv2.waitKey(0)
#     else:
#         print("Error: Could not read the first frame.")

# # Release resources
# cap.release()
cv2.destroyAllWindows()
