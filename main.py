import cv2
import numpy as np

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
    blurred = cv2.GaussianBlur(frame, (9,9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detected = False
    detected_name = ""

    # ------------------ Color Ranges ------------------

    # Brown (Potato)
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([25, 255, 180])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Red (Tomato) – 2 ranges
    lower_red1 = np.array([0, 150, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 150, 120])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # Onion color range (yellow-brown)
    lower_onion = np.array([10, 80, 80])
    upper_onion = np.array([25, 255, 255])

    mask_onion = cv2.inRange(hsv, lower_onion, upper_onion)

    # Clean masks
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel)

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, kernel)

    # ------------------ Potato Detection ------------------
    contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_brown:
        area = cv2.contourArea(cnt)
        if area > 15000:

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            if 0.6 < aspect_ratio < 1.4 and 0.45 < circularity < 0.85:
                detected = True
                detected_name = "Potato"
                break

    # ------------------ Tomato Detection ------------------
    if not detected:  # Only check tomato if potato not found
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 15000:

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                if 0.8 < aspect_ratio < 1.2 and circularity > 0.75:
                    detected = True
                    detected_name = "Tomato"
                    break

    if not detected:
        contours_onion, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_onion:
            area = cv2.contourArea(cnt)
            if area > 15000:

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                # Onion shape logic
                if 0.7 < aspect_ratio < 1.3 and 0.55 < circularity < 0.9:
                    detected_name = "Onion"
                    detected = True

    # ------------------ UI Section ------------------
    if detected:

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,215,255), 3)

        panel_x1 = x
        panel_y1 = y - 140 if y - 140 > 0 else y + h
        panel_x2 = x + 320
        panel_y2 = panel_y1 + 130

        cv2.rectangle(frame, (panel_x1, panel_y1),
                      (panel_x2, panel_y2),
                      (0,215,255), -1)

        cv2.putText(frame, detected_name.upper(),
                    (panel_x1 + 10, panel_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,0,0), 2)

        if detected_name == "Potato":
            cv2.putText(frame, "Calories : 77 kcal",
                        (panel_x1 + 10, panel_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Vitamin  : C, B6",
                        (panel_x1 + 10, panel_y1 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Benefits : Energy booster",
                        (panel_x1 + 10, panel_y1 + 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Good for digestion",
                        (panel_x1 + 10, panel_y1 + 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

        elif detected_name == "Tomato":
            cv2.putText(frame, "Calories : 18 kcal",
                        (panel_x1 + 10, panel_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Vitamin  : A, C",
                        (panel_x1 + 10, panel_y1 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Benefits : Good for heart",
                        (panel_x1 + 10, panel_y1 + 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Rich in antioxidants",
                        (panel_x1 + 10, panel_y1 + 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

    # Title
    cv2.putText(frame, "Smart Scan for Healthy Vegetables",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,0), 2)

    cv2.imshow("Smart Scan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
