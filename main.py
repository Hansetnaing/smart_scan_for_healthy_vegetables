import cv2
import numpy as np
from vegetable_data import vegetable_info

stable_name = ""
stable_count = 0
CONFIRM_FRAMES = 5

# ------------------ Sharp Detection ------------------

def count_sharp_ends(cnt):
    if len(cnt) > 150:
        cnt = cv2.approxPolyDP(cnt, 5, True)

    pts = cnt.reshape(-1,2)
    max_dist = 0
    p1 = p2 = None

    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            dist = np.linalg.norm(pts[i]-pts[j])
            if dist > max_dist:
                max_dist = dist
                p1 = pts[i]
                p2 = pts[j]

    sharp_count = 0

    for p in [p1,p2]:
        dists = np.linalg.norm(pts - p, axis=1)
        idx = np.argmin(dists)

        prev = pts[(idx-5)%len(pts)]
        nextp = pts[(idx+5)%len(pts)]

        v1 = prev - p
        v2 = nextp - p

        angle = abs(np.degrees(
            np.arctan2(v2[1],v2[0]) -
            np.arctan2(v1[1],v1[0])
        ))

        if angle < 80:
            sharp_count += 1

    return sharp_count

# ------------------ Camera ------------------

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5), np.uint8)

cv2.namedWindow("Smart Scan", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    box = 300   # smaller = faster
    sx = w//2 - box//2
    sy = h//2 - box//2
    ex = sx + box
    ey = sy + box

    cv2.rectangle(frame,(sx,sy),(ex,ey),(255,255,255),2)

    roi = frame[sy:ey, sx:ex]
    roi = cv2.convertScaleAbs(roi, alpha=1.1, beta=10)

    blur = cv2.GaussianBlur(roi,(5,5),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    roi_area = box * box
    detected = False
    detected_name = ""
    box_data = None

    # ------------------ MASKS ------------------

    mask_potato = cv2.inRange(hsv, (18, 40, 80), (35, 200, 255))
    mask_brown = cv2.inRange(hsv, (5, 50, 50), (20, 200, 255))
    mask_red    = cv2.inRange(hsv,(0,100,80),(10,255,255)) + \
                  cv2.inRange(hsv,(160,100,80),(179,255,255))
    mask_orange = cv2.inRange(hsv,(8,150,120),(25,255,255))
    mask_green  = cv2.inRange(hsv,(30,40,40),(90,255,255))

    mask_green  = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green  = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    def detect(mask):
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if not (8000 < area < roi_area*0.95):
            return None

        peri = cv2.arcLength(cnt,True)
        if peri == 0:
            return None

        circularity = 4*np.pi*area/(peri*peri)
        x_,y_,w_,h_ = cv2.boundingRect(cnt)
        ar = float(w_)/h_

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None

        solidity = area/hull_area
        sharp = count_sharp_ends(cnt)

        thickness = area / (w_ * h_)

        return area,circularity,ar,solidity,sharp,thickness,x_,y_,w_,h_

    # ------------------ POTATO ------------------

    data = detect(mask_potato | mask_brown)
    if data:
        area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data
        if 0.7 < ar < 1.5 and 0.55 < cir < 0.85 and sol > 0.90:
            detected = True
            detected_name = "Potato"
            box_data = (x_,y_,w_,h_)

    # ------------------ TOMATO (RED + ORANGE) ------------------

    if not detected:
        combined_mask = cv2.bitwise_or(mask_red, mask_orange)
        data = detect(combined_mask)
        if data:
            area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data
            if 0.8 < ar < 1.3 and cir > 0.75:
                detected = True
                detected_name = "Tomato"
                box_data = (x_,y_,w_,h_)

    # ------------------ CARROT ------------------

    if not detected:
        data = detect(mask_orange)
        if data:
            area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data
            if ar > 3.5:
                detected = True
                detected_name = "Carrot"
                box_data = (x_,y_,w_,h_)

    # ------------------ GREEN ------------------

    if not detected:
        data = detect(mask_green)
        if data:
            area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data

            if 0.8 < ar < 1.3 and cir > 0.75:
                detected_name = "Lime"

            elif ar > 5 and sharp >= 1:
                detected_name = "Ladyfinger"

            elif 3.3 < ar < 4.5 and thickness > 0.68:
                detected_name = "Cucumber"

            elif 2.5 < ar <= 4.2 and sharp >= 1 and sol < 0.93:
                detected_name = "Chili Green"

            elif area > 15000 and cir < 0.60 and sol < 0.85:
                detected_name = "Lettuce"

            print("AR:", ar, "SHARP:", sharp, "THICK:", thickness, "CIR:", cir, "Name", detected_name)

            if detected_name:
                detected = True
                box_data = (x_,y_,w_,h_)

    # ------------------ Stability ------------------

    if detected:
        if detected_name == stable_name:
            stable_count += 1
        else:
            stable_name = detected_name
            stable_count = 1
    else:
        stable_name = ""
        stable_count = 0

    # ------------------ Display ------------------

    if stable_count >= CONFIRM_FRAMES and box_data:

        x_,y_,w_,h_ = box_data
        x = x_ + sx
        y = y_ + sy

        cv2.rectangle(frame,(x,y),(x+w_,y+h_),(0,215,255),3)

        info = vegetable_info.get(stable_name)

        panel_y = y-150 if y-150>0 else y+h_
        cv2.rectangle(frame,(x,panel_y),(x+380,panel_y+140),(0,215,255),-1)

        cv2.putText(frame,stable_name.upper(),(x+10,panel_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

        cv2.putText(frame,"Calories: "+info["calories"],
                    (x+10,panel_y+45),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv2.putText(frame,"Nutrients: "+info["nutrient"],
                    (x+10,panel_y+65),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv2.putText(frame,"- "+info["benefit1"],
                    (x+20,panel_y+100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv2.putText(frame,"- "+info["benefit2"],
                    (x+20,panel_y+120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    cv2.imshow("Smart Scan",frame)
    cv2.imshow("Potato",mask_potato)
    cv2.imshow("Brown",mask_brown)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
