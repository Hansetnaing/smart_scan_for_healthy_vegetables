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

    box = 300
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

    mask_potato = cv2.inRange(hsv, (10, 30, 50), (45, 255, 255))
    mask_red    = cv2.inRange(hsv,(0,100,80),(10,255,255)) + \
                  cv2.inRange(hsv,(160,100,80),(179,255,255))
    mask_orange = cv2.inRange(hsv,(8,120,120),(30,255,255))
    mask_green  = cv2.inRange(hsv,(30,40,40),(90,255,255))

    mask_green  = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green  = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    mask_lettuce = cv2.inRange(hsv, (30, 40, 50), (65, 255, 255))
    mask_lettuce = cv2.morphologyEx(mask_lettuce, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

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

    data = detect(mask_potato)
    if data:
        area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data
        if 0.70 < ar < 1.45 and 0.70 < cir < 0.85 and sol > 0.85 and thickness > 0.70:
            detected = True
            detected_name = "Potato"
            box_data = (x_,y_,w_,h_)

    # ------------------ RED ------------------

    if not detected:
        data = detect(mask_red)
        if data:
            area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data

            if 0.85 < ar < 1.2 and cir > 0.80 and sol > 0.90:
                detected_name = "Tomato"
                print("Name: ",detected_name,"Ar: ",ar,"Cir: ",cir,"Sol: ",sol)

            elif ar > 2.5 and thickness > 0.6:
                detected_name = "Eggplant"

            if detected_name:
                detected = True
                box_data = (x_,y_,w_,h_)

            print("Name: ",detected_name)

    # ------------------ ORANGE ------------------

    if not detected:
        data = detect(mask_orange)
        if data:
            area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data

            if ar > 3.5:
                detected_name = "Carrot"

            elif 2.0 < ar < 3.5 and thickness > 0.65:
                detected_name = "Corn"

            if detected_name:
                detected = True
                box_data = (x_,y_,w_,h_)

            print("Name: ",detected_name)



    # ------------------ GREEN ------------------

    if not detected:
        data = detect(mask_green)
        if data:
            area,cir,ar,sol,sharp,thickness,x_,y_,w_,h_ = data
            detected_name = ""

            # Ladyfinger
            if ar > 5 and sharp >= 1:
                detected_name = "Ladyfinger"

            # Cucumber
            elif 3.5 < ar < 4.5 and thickness > 0.75 and sol > 0.92:
                detected_name = "Cucumber"

            # Chili
            elif 2.5 < ar <= 4.2 and sharp >= 1 and sol < 0.93:
                detected_name = "Chili Green"

            # Lime
            elif 0.8 < ar < 1.3 and cir > 0.75 and sol > 0.85:
                detected_name = "Lime"

            print("Detected:", detected_name)

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

    if stable_count >= CONFIRM_FRAMES and box_data and stable_name in vegetable_info:

        x_, y_, w_, h_ = box_data
        x = x_ + sx
        y = y_ + sy

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 215, 255), 3)

        info = vegetable_info.get(stable_name)

        # ---------- Side Panel ----------
        panel_x = frame.shape[1] - 420
        panel_y = 40
        panel_width = 380

        # Collect all benefits automatically
        benefits = [v for k, v in info.items() if k.startswith("benefit")]

        panel_height = 120 + (len(benefits) * 25)

        # Transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 215, 255), -1)

        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # ---------- Text ----------
        cv2.putText(frame, stable_name.upper(),
                    (panel_x + 15, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.putText(frame, "Calories: " + info["calories"],
                    (panel_x + 15, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        cv2.putText(frame, "Nutrients: " + info["nutrient"],
                    (panel_x + 15, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        y_text = panel_y + 110
        for b in benefits:
            cv2.putText(frame, "- " + b,
                        (panel_x + 25, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_text += 25

    # ------------------ ERROR PANEL ------------------

    else:

        panel_x = frame.shape[1] - 420
        panel_y = 40
        panel_width = 380
        panel_height = 100

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 255), -1)

        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "VEGETABLE NOT RECOGNIZED",
                    (panel_x + 20, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Place a valid vegetable inside the box",
                    (panel_x + 20, panel_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)



    cv2.imshow("Smart Scan",frame)
    cv2.imshow("Green",mask_green)
    cv2.imshow("Red",mask_red)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
