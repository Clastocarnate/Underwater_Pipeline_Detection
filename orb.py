import cv2

# Initialize the video capture object
cap = cv2.VideoCapture('WhatsApp Video 2024-04-21 at 10.01.17 PM.mp4')

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Initialize ORB detector
orb = cv2.ORB_create()

while True:
    # Read a new frame
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Crop the frame by 100 pixels from each side
    # Check if frame is large enough to be cropped
    # if frame.shape[1] > 200 and frame.shape[0] > 200:
    #     frame = frame[200:-200, 200:-200]  # Crop 100 pixels from all sides
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features in the frame
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Draw keypoints on the frame
    output = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    
    # Display the frame with keypoints
    cv2.imshow('Frame with ORB features', output)
    
    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
