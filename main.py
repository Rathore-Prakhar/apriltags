import cv2
from pupil_apriltags import Detector
import math
import numpy as np
import threading

validTagIds = [4, 7, 3, 8]

aprilTagX = [0-14/12.0]*20
aprilTagY = [6-6/12.0]*20
aprilAngle = [180]*20
aprilHeights = [10]*20

aprilTagX[4] = -104.4/12  # RED
aprilTagY[4] = 0.0/12.0
aprilAngle[4] = 0
aprilHeights[4] = 53.88

aprilTagX[3] = -126.65/12  # RED
aprilTagY[3] = 0/12.0
aprilAngle[3] = 0
aprilHeights[3] = 53.88

aprilTagX[7] = -104.4/12  # BLUE
aprilTagY[7] = 651.25/12.0
aprilAngle[7] = 0
aprilHeights[7] = 53.88

aprilTagX[8] = -127.08/12  # BLUE
aprilTagY[8] = 651.25/12
aprilAngle[8] = 0
aprilHeights[8] = 53.88

# Camera params
cameraMatrix = np.array([(336.7755634193813, 0.0, 333.3575643300718), 
                         (0.0, 336.02729840829176, 212.77376312080065), 
                         (0.0, 0.0, 1.0)])
camera_params = (cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2])

class FrameGrabber(threading.Thread):
    def __init__(self, src=0):
        super(FrameGrabber, self).__init__()
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def stop(self):
        self.running = False
        self.cap.release()

def calculate_distance_to_tag(tag_center, frame_shape, tag_height):
    f = 587.4786864517579
    mountHeight = 12
    mountAngle = 37 * (math.pi/180)
    x = tag_center[0] - frame_shape[1]/2
    y = frame_shape[0]/2 - tag_center[1]
    VertAngle = mountAngle + math.atan(y/f)
    yDist = (tag_height - mountHeight) / math.tan(VertAngle)
    xDist = ((tag_height - mountHeight) / math.sin(VertAngle)) * x / (math.sqrt(f*f + y*y))
    return (xDist, yDist, tag_height)

def main():
    frame_grabbers = [FrameGrabber(src=i) for i in range(3)]
    for fg in frame_grabbers:
        fg.start()

    at_detector = Detector(families='tag36h11',
                           nthreads=4,  # need to test on pi
                           quad_decimate=2.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    while True:
        frames = []
        for fg in frame_grabbers:
            if not fg.ret:
                continue
            frames.append(fg.frame)
        
        if not frames:
            continue
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = at_detector.detect(gray)
            frame_height, frame_width = frame.shape[:2]

            for r in results:
                if r.tag_id in validTagIds:
                    ptUpperL, ptUpperR, ptBottomL, ptBottomR = r.corners
                    ptUpperL = (int(ptUpperL[0]), int(ptUpperL[1]))
                    ptUpperR = (int(ptUpperR[0]), int(ptUpperR[1]))
                    ptBottomL = (int(ptBottomL[0]), int(ptBottomL[1]))
                    ptBottomR = (int(ptBottomR[0]), int(ptBottomR[1]))
                    
                    cv2.line(frame, ptUpperL, ptUpperR, (0, 255, 0), 2)
                    cv2.line(frame, ptUpperR, ptBottomR, (0, 255, 0), 2)
                    cv2.line(frame, ptBottomR, ptBottomL, (0, 255, 0), 2)
                    cv2.line(frame, ptBottomL, ptUpperL, (0, 255, 0), 2)
                    
                    cX, cY = int(r.center[0]), int(r.center[1])
                    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                    
                    distance_to_tag = calculate_distance_to_tag(r.center, frame.shape, aprilHeights[r.tag_id])
                    cv2.putText(frame, f"Distance: {distance_to_tag[0]:.2f}, {distance_to_tag[1]:.2f}, {distance_to_tag[2]:.2f}", 
                                (ptUpperL[0], ptUpperL[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for i, frame in enumerate(frames):
            cv2.imshow(f'AprilTag Detection Camera {i+1}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for fg in frame_grabbers:
        fg.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
