import sys
import os
import face_recognition
import cv2
import dlib
import numpy as np
from datetime import datetime, timedelta
import csv
import subprocess
import fil1
import os



def run_registration_program():
    registration_program = "python3 register.py"  # Replace with the command to run your register.py program
    subprocess.Popen(registration_program, shell=True)
# Initialize dlib face detector and facial landmark predictor
def run_recognition():
    
    file_path = '/home/parthjoshi/Desktop/webside/face_detection/Face_recognition/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(file_path)
   

    # Initialize lists for known face encodings and names
    previous_landmarks = None
    liveness_threshold = 5
    known_faces = []
    known_names = []

    # Directory containing face images
    directory = fil1.faces_directory  # Replace with the path to your directory

    # Load known faces and encodings from directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            # Detect face landmarks
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            if len(face_locations) > 0:
                face_location = face_locations[0]
                top, right, bottom, left = face_location

                # Perform face encoding
                face_encoding = face_recognition.face_encodings(rgb, [face_location])[0]

                # Add face encoding and name to the lists
                known_faces.append(face_encoding)
                known_names.append(name)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Replace with your video source if not using the webcam

    # Get screen resolution
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Calculate ROI dimensions
    roi_width = int(screen_width / 2)
    roi_height = int(screen_height / 2)
    roi_x = int((screen_width - roi_width) / 2)
    roi_y = int((screen_height - roi_height) / 2)

    # Initialize tolerance for face recognition (confidence factor)
    tolerance = 0.5  # Adjust this value as needed

    # Initialize dictionary to track last detected time for recognized faces and unregistered faces
    recognized_last_detected = {}
    unregistered_last_detected = {}

    # Create resizable window
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", int(screen_width), int(screen_height))

    # Initialize variables for motion detection
    prev_frame_gray = None
    stationary_counter = 0

    # Main loop
    while True:
        # Read a frame
        ret, frame = cap.read()

        # Convert frame to gray for motion detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

        # Initialize previous frame if it's the first iteration
        if prev_frame_gray is None:
            prev_frame_gray = frame_gray
            continue

        # Calculate absolute difference between current and previous frames
        frame_delta = cv2.absdiff(prev_frame_gray, frame_gray)

        # Apply threshold to detect significant intensity differences
        threshold = 30
        _, frame_threshold = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)

        # Dilate the thresholded image to fill in holes
        kernel = np.ones((5, 5), np.uint8)
        frame_threshold = cv2.dilate(frame_threshold, kernel, iterations=2)

        # Find contours of significant differences
        contours, _ = cv2.findContours(frame_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert frame back to RGB for face recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw ROI on the frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        # Initialize set to track unregistered face IDs
        unregistered_faces = set()

        # Iterate over the face encodings
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Check if the face is completely outside the ROI
            if right < roi_x or left > roi_x + roi_width or bottom < roi_y or top > roi_y + roi_height:
                # Draw bounding box in gray color
                cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 2)
                continue

            # Compare face encodings with known faces
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            face_distance = face_distances[best_match_index]

            if face_distance <= tolerance:
                name = known_names[best_match_index]
                current_time = datetime.now()

                if name not in recognized_last_detected or (current_time - recognized_last_detected[name]) >= timedelta(minutes=2):
                    recognized_last_detected[name] = current_time
                    record_attendance(name, current_time)
                else:
                    unregistered_faces.add(name)
                
                landmarks = predictor(rgb, dlib.rectangle(left, top, right, bottom))
                

                # Draw bounding box and display name on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw bounding box for unknown face in gray color within ROI
                cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 2)
                cv2.putText(frame, "Unregistered", (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)
                unregistered_faces.add("Unregistered")

        # Record attendance for unregistered faces if there is motion
        current_time = datetime.now()
        motion_detected = len(contours) > 0
        if motion_detected:
            stationary_counter = 0
            for name in unregistered_faces:
                if name not in unregistered_last_detected or (current_time - unregistered_last_detected[name]) >= timedelta(minutes=1):
                    unregistered_last_detected[name] = current_time
                    record_attendance(name, current_time)
        else:
            stationary_counter += 1
            if stationary_counter >= 30:
                unregistered_faces.clear()

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Check for keyboard interrupt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Check for 'r' key press
            run_registration_program()




    # Release the video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

def record_attendance(name, current_time):
    date_str = current_time.strftime("%Y-%m-%d")
    csv_filename = f"{date_str}.csv"

    # Check if the CSV file already exists
    if not os.path.isfile(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time"])

    # Append the attendance entry to the CSV file
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_time.strftime("%H:%M:%S")])

if __name__ == "__main__":
    run_recognition()
