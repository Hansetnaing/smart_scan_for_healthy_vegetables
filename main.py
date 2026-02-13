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

    # ------------------ Scanning Area (Center Box) ------------------
    height, width, _ = frame.shape

    box_w = 400
    box_h = 400

    start_x = width // 2 - box_w // 2
    start_y = height // 2 - box_h // 2
    end_x = start_x + box_w
    end_y = start_y + box_h

    # Draw white border
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255,255,255), 2)

    # Crop ROI
    roi = frame[start_y:end_y, start_x:end_x]

    blurred = cv2.GaussianBlur(roi, (9,9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detected = False
    detected_name = ""

    # ------------------ Color Ranges ------------------

    # Potato (Brown)
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([25, 255, 180])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Tomato (Red - 2 ranges)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # Garlic (White / Cream - stricter)
    lower_garlic = np.array([0, 10, 150])
    upper_garlic = np.array([40, 120, 255])

    mask_garlic = cv2.inRange(hsv, lower_garlic, upper_garlic)

    # Clean masks
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel)

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    mask_garlic = cv2.morphologyEx(mask_garlic, cv2.MORPH_CLOSE, kernel)
    mask_garlic = cv2.morphologyEx(mask_garlic, cv2.MORPH_OPEN, kernel)

    roi_area = box_w * box_h

    # ------------------ Potato Detection ------------------
    contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_brown:
        area = cv2.contourArea(cnt)
        if 10000 < area < roi_area * 0.8:

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = float(w) / h

            if 0.6 < aspect_ratio < 1.4 and 0.45 < circularity < 0.85:
                x = x + start_x
                y = y + start_y
                detected = True
                detected_name = "Potato"
                break

    # ------------------ Tomato Detection ------------------
    if not detected:
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if 10000 < area < roi_area * 0.8:

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(cnt)

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue

                solidity = float(area) / hull_area
                aspect_ratio = float(w) / h

                if 0.75 < aspect_ratio < 1.3 and circularity > 0.65 and solidity > 0.85:
                    x = x + start_x
                    y = y + start_y
                    detected = True
                    detected_name = "Tomato"
                    break

    # ------------------ Garlic Detection ------------------
    if not detected:
        contours_garlic, _ = cv2.findContours(mask_garlic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_garlic:
            area = cv2.contourArea(cnt)
            if 8000 < area < roi_area * 0.6:

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(cnt)

                # Prevent border touching
                if x <= 5 or y <= 5 or x+w >= box_w-5 or y+h >= box_h-5:
                    continue

                aspect_ratio = float(w) / h

                if 0.6 < aspect_ratio < 1.4 and circularity > 0.5:
                    x = x + start_x
                    y = y + start_y
                    detected = True
                    detected_name = "Garlic"
                    break

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

        elif detected_name == "Garlic":
            cv2.putText(frame, "Calories : 149 kcal",
                        (panel_x1 + 10, panel_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Vitamin  : B6, C",
                        (panel_x1 + 10, panel_y1 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            cv2.putText(frame, "Benefits : Boost immunity",
                        (panel_x1 + 10, panel_y1 + 95),
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
