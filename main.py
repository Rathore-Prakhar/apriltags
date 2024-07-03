import cv2
from pupil_apriltags import Detector
import math
import numpy as np

validTagIds = [1, 2]

# arbitrary values
camera_height = 5.0
tag_height = 10.0

def calculate_distance_to_tag(camera_height, tag_height, tag_y_position, frame_height, fov_degrees):
    fov_radians = math.radians(fov_degrees)
    frame_pixel_height = 2 * math.tan(fov_radians / 2) * (camera_height - tag_height)
    angle_of_depression = math.atan2((frame_height / 2 - tag_y_position), frame_pixel_height)
    horizontal_distance = (tag_height - camera_height) / math.tan(angle_of_depression)
    return horizontal_distance

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    at_detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = at_detector.detect(gray)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        for r in results:
            if r.tag_id in validTagIds:
                (ptUpperL, ptUpperR, ptBottomL, ptBottomR) = r.corners
                ptUpperL = (int(ptUpperL[0]), int(ptUpperL[1]))
                ptUpperR = (int(ptUpperR[0]), int(ptUpperR[1]))
                ptBottomL = (int(ptBottomL[0]), int(ptBottomL[1]))
                ptBottomR = (int(ptBottomR[0]), int(ptBottomR[1]))
                cv2.line(frame, ptUpperL, ptUpperR, (0, 255, 0), 2)
                cv2.line(frame, ptUpperR, ptBottomR, (0, 255, 0), 2)
                cv2.line(frame, ptBottomR, ptBottomL, (0, 255, 0), 2)
                cv2.line(frame, ptBottomL, ptUpperL, (0, 255, 0), 2)
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                vertical_fov_degrees = 60.0
                distance_to_tag = calculate_distance_to_tag(camera_height, tag_height, cY, frame_height, vertical_fov_degrees)
                cv2.putText(frame, f"Distance: {distance_to_tag:.2f} feet", (ptUpperL[0], ptUpperL[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('AprilTag Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
