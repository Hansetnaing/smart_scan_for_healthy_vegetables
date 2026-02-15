import cv2
import numpy as np

# ------------------ Stability ------------------

stable_name = ""
stable_count = 0
CONFIRM_FRAMES = 5

# ------------------ Vegetable Info ------------------

vegetable_info = {
    "Potato": {"calories": "77 kcal","nutrient": "Vitamin C, B6, Potassium","benefit1": "Energy booster","benefit2": "Good for digestion"},
    "Tomato": {"calories": "18 kcal","nutrient": "Vitamin A, C, Lycopene","benefit1": "Good for heart","benefit2": "Rich in antioxidants"},
    "Cucumber": {"calories": "16 kcal","nutrient": "Vitamin K","benefit1": "Hydrates body","benefit2": "Good for skin"},
    "Ladyfinger": {"calories": "33 kcal","nutrient": "Fiber","benefit1": "Improves digestion","benefit2": "Controls blood sugar"},
    "Lettuce": {"calories": "15 kcal","nutrient": "Vitamin A, K","benefit1": "Good for skin","benefit2": "Supports hydration"},
    "Carrot": {"calories": "41 kcal","nutrient": "Vitamin A (Beta-carotene)","benefit1": "Improves eyesight","benefit2": "Boosts immunity"},
    "Chili Red": {"calories": "40 kcal","nutrient": "Vitamin C","benefit1": "Boosts metabolism","benefit2": "Improves circulation"},
    "Chili Green": {"calories": "30 kcal","nutrient": "Vitamin C","benefit1": "Improves digestion","benefit2": "Rich in antioxidants"},
    "Lime": {"calories": "30 kcal","nutrient": "Vitamin C","benefit1": "Boosts immunity","benefit2": "Improves digestion"}
}

# ------------------ Tip Detection Function ------------------

def count_sharp_ends(cnt):

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

    box = 400
    sx = w//2 - box//2
    sy = h//2 - box//2
    ex = sx + box
    ey = sy + box

    cv2.rectangle(frame, (sx, sy), (ex, ey), (255,255,255), 2)

    roi = frame[sy:ey, sx:ex]
    roi = cv2.convertScaleAbs(roi, alpha=1.1, beta=10)

    blur = cv2.GaussianBlur(roi, (9,9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    roi_area = box * box
    detected = False
    detected_name = ""
    box_data = None

    # ------------------ COLOR MASKS ------------------

    mask_brown = cv2.inRange(hsv,(8,80,30),(25,255,200))
    mask_red = cv2.inRange(hsv,(0,100,80),(10,255,255)) + \
               cv2.inRange(hsv,(160,100,80),(179,255,255))
    mask_orange = cv2.inRange(hsv,(8,150,120),(25,255,255))
    mask_green = cv2.inRange(hsv,(30,40,40),(90,255,255))

    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    def detect(mask):
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 8000 < area < roi_area*0.95:
                peri = cv2.arcLength(cnt,True)
                if peri==0: continue
                circularity = 4*np.pi*area/(peri*peri)
                x_,y_,w_,h_ = cv2.boundingRect(cnt)
                ar = float(w_)/h_
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area==0: continue
                solidity = area/hull_area
                sharp = count_sharp_ends(cnt)
                return area,circularity,ar,solidity,sharp,x_,y_,w_,h_
        return None,None,None,None,None,None,None,None,None

    # ------------------ Detection ------------------

    area,cir,ar,sol,sharp,x_,y_,w_,h_ = detect(mask_green)

    if area:

        # Lime
        if 0.75 < ar < 1.35 and cir > 0.75:
            detected = True
            detected_name = "Lime"

        # Chili Green
        elif ar > 4.0 and sharp >= 1:
            detected = True
            detected_name = "Chili Green"

        # Ladyfinger
        elif 2.8 < ar <= 4.0 and sharp == 1:
            detected = True
            detected_name = "Ladyfinger"

        # Cucumber
        elif 2.0 < ar <= 3.5 and sharp == 0:
            detected = True
            detected_name = "Cucumber"

        # Lettuce
        elif area > 20000 and cir < 0.60:
            detected = True
            detected_name = "Lettuce"

        if detected:
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

        cv2.putText(frame,"Benefits:",
                    (x+10,panel_y+85),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv2.putText(frame,"- "+info["benefit1"],
                    (x+20,panel_y+105),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv2.putText(frame,"- "+info["benefit2"],
                    (x+20,panel_y+125),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    cv2.imshow("Smart Scan",frame)
    cv2.imshow("Green Mask",mask_red)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
