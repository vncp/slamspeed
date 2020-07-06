import time
import cv2
from display import Display
from extractor import Extractor
import numpy as np

def process_frame(img):
    matches = fe.extract(img)
    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1.pt)
        u2, v2 = map(lambda x: int(round(x)), pt2.pt)
        cv2.circle(img, (u1,v1), 2, (10,10, 200))
        cv2.line(img, (u1, v1), (u2, v2), (255,0,0))
    disp.draw(img)

disp = Display(640, 480)
fe = Extractor()
if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break