import cv2
import numpy as np

# ------------------ Stability System ------------------

stable_name = ""
stable_count = 0
CONFIRM_FRAMES = 8

# ------------------ Vegetable Information ------------------

vegetable_info = {

    "Potato": {
        "calories": "77 kcal",
        "nutrient": "Vitamin C, B6, Potassium",
        "benefit1": "Energy booster",
        "benefit2": "Good for digestion"
    },

    "Tomato": {
        "calories": "18 kcal",
        "nutrient": "Vitamin A, C, Lycopene",
        "benefit1": "Good for heart",
        "benefit2": "Rich in antioxidants"
    },

    "Carrot": {
        "calories": "41 kcal",
        "nutrient": "Vitamin A",
        "benefit1": "Good for eyes",
        "benefit2": "Boosts immunity"
    },

    "Cucumber": {
        "calories": "16 kcal",
        "nutrient": "Vitamin K",
        "benefit1": "Hydrates body",
        "benefit2": "Good for skin"
    },

    "Ladyfinger": {
        "calories": "33 kcal",
        "nutrient": "Fiber",
        "benefit1": "Improves digestion",
        "benefit2": "Controls blood sugar"
    },

    "Chili": {
        "calories": "40 kcal",
        "nutrient": "Vitamin C, Capsaicin",
        "benefit1": "Boosts metabolism",
        "benefit2": "Improves immunity"
    },

    "Eggplant": {
        "calories": "25 kcal",
        "nutrient": "Fiber, Antioxidants",
        "benefit1": "Good for brain",
        "benefit2": "Improves digestion"
    },

    "Cabbage": {
        "calories": "25 kcal",
        "nutrient": "Vitamin C, K",
        "benefit1": "Boosts immunity",
        "benefit2": "Good for gut health"
    },

    "Lettuce": {
        "calories": "15 kcal",
        "nutrient": "Vitamin A, K",
        "benefit1": "Hydrates body",
        "benefit2": "Good for skin"
    }
}

# ------------------ Camera Setup ------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Smart Scan", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Scan", 1280, 720)

kernel = np.ones((5,5), np.uint8)

# ------------------ Main Loop ------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # ROI box
    box_w = 400
    box_h = 400
    start_x = width // 2 - box_w // 2
    start_y = height // 2 - box_h // 2
    end_x = start_x + box_w
    end_y = start_y + box_h

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255,255,255), 2)

    roi = frame[start_y:end_y, start_x:end_x]

    blurred = cv2.GaussianBlur(roi, (9,9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detected = False
    detected_name = ""
    box_data = None
    roi_area = box_w * box_h

    # ------------------ COLOR MASKS ------------------

    mask_brown = cv2.inRange(hsv, np.array([10,60,20]), np.array([25,255,180]))

    mask_red = cv2.inRange(hsv, np.array([0,100,80]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([160,100,80]), np.array([179,255,255]))

    mask_orange = cv2.inRange(hsv, np.array([8,150,120]), np.array([25,255,255]))

    mask_green = cv2.inRange(hsv, np.array([35,80,60]), np.array([85,255,255]))

    mask_purple = cv2.inRange(hsv, np.array([125,50,50]), np.array([155,255,255]))

    mask_light_green = cv2.inRange(hsv, np.array([40,40,80]), np.array([75,200,255]))

    # Myanmar Pumpkin (dark green)
    mask_pumpkin_mm = cv2.inRange(
        hsv,
        np.array([35, 40, 40]),
        np.array([75, 200, 200])
    )

    masks = [mask_brown, mask_red, mask_orange,
             mask_green, mask_purple,
             mask_light_green, mask_pumpkin_mm]

    masks = [cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel) for m in masks]

    # ------------------ DETECTION FUNCTION ------------------

    def detect(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 8000 < area < roi_area * 0.9:
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circularity = 4 * np.pi * area / (peri * peri)
                x_, y_, w_, h_ = cv2.boundingRect(cnt)
                ar = float(w_) / h_
                return area, circularity, ar, x_, y_, w_, h_
        return None, None, None, None, None, None, None

    # ------------------ DETECTION LOGIC ------------------

    # Potato
    area, cir, ar, x_, y_, w_, h_ = detect(mask_brown)
    if area and 0.6 < ar < 1.4:
        detected = True
        detected_name = "Potato"
        box_data = (x_, y_, w_, h_)

    # Tomato
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_red)
        if area and cir > 0.65 and 0.8 < ar < 1.2:
            detected = True
            detected_name = "Tomato"
            box_data = (x_, y_, w_, h_)

    # Chili (long red or green)
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_red | mask_green)
        if area and ar > 2.5:
            detected = True
            detected_name = "Chili"
            box_data = (x_, y_, w_, h_)

    # Carrot
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_orange)
        if area and ar > 2.5:
            detected = True
            detected_name = "Carrot"
            box_data = (x_, y_, w_, h_)

    # Eggplant
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_purple)
        if area and ar > 1.5:
            detected = True
            detected_name = "Eggplant"
            box_data = (x_, y_, w_, h_)

    # Cabbage
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_light_green)
        if area and cir > 0.7:
            detected = True
            detected_name = "Cabbage"
            box_data = (x_, y_, w_, h_)

    # Lettuce
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_green)
        if area and cir > 0.5:
            detected = True
            detected_name = "Lettuce"
            box_data = (x_, y_, w_, h_)

    # ------------------ STABILITY ------------------

    if detected:
        if detected_name == stable_name:
            stable_count += 1
        else:
            stable_name = detected_name
            stable_count = 1
    else:
        stable_name = ""
        stable_count = 0

    # ------------------ DRAW RESULT ------------------

    if stable_count >= CONFIRM_FRAMES and box_data:

        x_, y_, w_, h_ = box_data
        x = x_ + start_x
        y = y_ + start_y

        cv2.rectangle(frame, (x,y), (x+w_,y+h_), (0,215,255), 3)

        panel_y = y - 150 if y - 150 > 0 else y + h_

        cv2.rectangle(frame,
                      (x, panel_y),
                      (x+380, panel_y+140),
                      (0,215,255), -1)

        info = vegetable_info.get(stable_name)

        cv2.putText(frame, stable_name.upper(),
                    (x+10, panel_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,0), 2)

        cv2.putText(frame, "Calories: " + info["calories"],
                    (x+10, panel_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.putText(frame, "Nutrients: " + info["nutrient"],
                    (x+10, panel_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.putText(frame, "Benefit: " + info["benefit1"],
                    (x+10, panel_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.putText(frame, info["benefit2"],
                    (x+10, panel_y+115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.putText(frame, "Smart Scan for Healthy Vegetables",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,0), 2)

    cv2.imshow("Smart Scan", frame)
    cv2.imshow("Red Mask", mask_red)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
