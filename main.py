import cv2
import numpy as np
from vegetable_data import vegetable_info

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5), np.uint8)

stable_name = ""
stable_count = 0
CONFIRM_FRAMES = 5

cv2.namedWindow("Smart Scan", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    blur = cv2.GaussianBlur(frame, (7,7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    detected = False
    detected_name = ""
    box_data = None

    # -------- COLOR MASKS --------

    mask_green = cv2.inRange(hsv,(30,40,40),(90,255,255))
    mask_red1  = cv2.inRange(hsv,(0,70,50),(10,255,255))
    mask_red2  = cv2.inRange(hsv,(170,70,50),(180,255,255))
    mask_red = mask_red1 + mask_red2
    mask_orange = cv2.inRange(hsv,(10,100,100),(25,255,255))
    mask_yellow = cv2.inRange(hsv,(20,80,80),(35,255,255))
    mask_brown = cv2.inRange(hsv,(5,50,20),(20,255,200))

    masks = {
        "green": mask_green,
        "red": mask_red,
        "orange": mask_orange,
        "yellow": mask_yellow,
        "brown": mask_brown
    }

    # -------- PROCESS EACH COLOR --------

    for color, mask in masks.items():

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 8000:
            continue

        peri = cv2.arcLength(largest, True)
        circularity = 4*np.pi*area/(peri*peri) if peri!=0 else 0

        x,y,w,h = cv2.boundingRect(largest)
        ar = float(w)/h if h!=0 else 0
        thickness_ratio = min(w,h)/max(w,h)

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area/hull_area if hull_area!=0 else 0

        sharp_score = 0
        epsilon = 0.02*peri
        approx = cv2.approxPolyDP(largest, epsilon, True)

        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i+1)%len(approx)][0]
            p3 = approx[(i+2)%len(approx)][0]

            v1 = p2-p1
            v2 = p3-p2

            angle = abs(np.degrees(
                np.arctan2(v2[1],v2[0]) -
                np.arctan2(v1[1],v1[0])
            ))

            if angle > 45:
                sharp_score += 1

        # -------- CLASSIFICATION --------

        if color == "green":

            if ar > 4.5 and sharp_score >= 2:
                detected_name = "Chili Green"

            elif 2.5 < ar <= 4.5 and circularity > 0.55:
                detected_name = "Cucumber"

            elif 2.5 < ar <= 4.5 and solidity > 0.88:
                detected_name = "Ladyfinger"

            elif circularity > 0.75:
                detected_name = "Lettuce"

        elif color == "red":

            if circularity > 0.75:
                detected_name = "Tomato"
            else:
                detected_name = "Chili Red"

        elif color == "orange":
            detected_name = "Carrot"

        elif color == "yellow":
            detected_name = "Lime"

        elif color == "brown":
            detected_name = "Potato"

        if detected_name:
            detected = True
            box_data = (x,y,w,h)
            break

    # -------- Stability --------

    if detected:
        if detected_name == stable_name:
            stable_count += 1
        else:
            stable_name = detected_name
            stable_count = 1
    else:
        stable_name = ""
        stable_count = 0

    # -------- DISPLAY --------

    if stable_count >= CONFIRM_FRAMES and box_data:

        x,y,w,h = box_data
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,215,255),3)

        info = vegetable_info.get(stable_name)

        panel_y = y-120 if y-120>0 else y+h

        cv2.rectangle(frame,
                      (x,panel_y),
                      (x+400,panel_y+110),
                      (0,215,255),-1)

        cv2.putText(frame, stable_name.upper(),
                    (x+10,panel_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,0,0),2)

        cv2.putText(frame,"Calories: "+info["calories"],
                    (x+10,panel_y+50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,0,0),1)

        cv2.putText(frame,"Nutrients: "+info["nutrient"],
                    (x+10,panel_y+70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,0,0),1)

    cv2.imshow("Smart Scan", frame)
    cv2.imshow("Brown",mask_brown)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
