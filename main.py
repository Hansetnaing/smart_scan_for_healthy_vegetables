import cv2
import numpy as np

# ------------------ Vegetable Information ------------------

vegetable_info = {
    "Potato": {
        "calories": "77 kcal (per 100g)",
        "nutrient": "Vitamin C, B6, Potassium",
        "benefit1": "Energy booster",
        "benefit2": "Good for digestion"
    },
    "Tomato": {
        "calories": "18 kcal (per 100g)",
        "nutrient": "Vitamin A, C, Lycopene",
        "benefit1": "Good for heart",
        "benefit2": "Rich in antioxidants"
    },
    "Carrot": {
        "calories": "41 kcal (per 100g)",
        "nutrient": "Vitamin A (Beta-carotene)",
        "benefit1": "Good for eyes",
        "benefit2": "Boosts immunity"
    },
    "Cucumber": {
        "calories": "16 kcal (per 100g)",
        "nutrient": "Vitamin K, Magnesium",
        "benefit1": "Hydrates body",
        "benefit2": "Good for skin"
    },
    "Ladyfinger": {
        "calories": "33 kcal (per 100g)",
        "nutrient": "Fiber, Vitamin C, Folate",
        "benefit1": "Improves digestion",
        "benefit2": "Controls blood sugar"
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

    # -------- Center Scanning Box --------
    height, width, _ = frame.shape
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
    roi_area = box_w * box_h

    # -------- Color Ranges --------

    # Potato (Brown)
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([25, 255, 180])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Tomato (Red)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # Carrot (Orange)
    lower_carrot = np.array([8, 150, 120])
    upper_carrot = np.array([18, 255, 255])
    mask_carrot = cv2.inRange(hsv, lower_carrot, upper_carrot)

    # Green (Cucumber & Ladyfinger)
    lower_green = np.array([35, 80, 60])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Clean masks
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_carrot = cv2.morphologyEx(mask_carrot, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # -------- Detection --------

    def detect(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 8000 < area < roi_area * 0.9:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x_, y_, w_, h_ = cv2.boundingRect(cnt)
                aspect_ratio = float(w_) / h_

                return area, circularity, aspect_ratio, x_, y_, w_, h_
        return None, None, None, None, None, None, None

    # Potato
    area, cir, ar, x_, y_, w_, h_ = detect(mask_brown)
    if area and 0.6 < ar < 1.4 and 0.45 < cir < 0.85:
        detected = True
        detected_name = "Potato"

    # Tomato
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_red)
        if area and 0.75 < ar < 1.3 and cir > 0.65:
            detected = True
            detected_name = "Tomato"

    # Carrot
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_carrot)
        if area and (ar > 2.0 or ar < 0.5):
            detected = True
            detected_name = "Carrot"

    # Green (Cucumber / Ladyfinger)
    if not detected:
        area, cir, ar, x_, y_, w_, h_ = detect(mask_green)
        if area and (ar > 2.0 or ar < 0.5):
            if area > 20000:
                detected_name = "Cucumber"
            else:
                detected_name = "Ladyfinger"
            detected = True

    # -------- Draw Result --------
    if detected:

        x = x_ + start_x
        y = y_ + start_y
        w = w_
        h = h_

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,215,255), 3)

        panel_x1 = x
        panel_y1 = y - 150 if y - 150 > 0 else y + h

        cv2.rectangle(frame,
                      (panel_x1, panel_y1),
                      (panel_x1+350, panel_y1+140),
                      (0,215,255), -1)

        info = vegetable_info.get(detected_name)

        cv2.putText(frame, detected_name.upper(),
                    (panel_x1+10, panel_y1+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,0), 2)

        cv2.putText(frame, "Calories: " + info["calories"],
                    (panel_x1+10, panel_y1+55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 1)

        cv2.putText(frame, "Nutrients: " + info["nutrient"],
                    (panel_x1+10, panel_y1+75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 1)

        cv2.putText(frame, "Benefit: " + info["benefit1"],
                    (panel_x1+10, panel_y1+95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 1)

        cv2.putText(frame, info["benefit2"],
                    (panel_x1+10, panel_y1+115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 1)

    cv2.putText(frame, "Smart Scan for Healthy Vegetables",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,0), 2)

    cv2.imshow("Smart Scan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
