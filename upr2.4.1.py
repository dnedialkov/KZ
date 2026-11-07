import cv2
import numpy as np


def create_sample_image():
    # Генерираме тестово изображение с няколко фигури
    img = np.zeros((400, 500, 3), dtype=np.uint8)

    # Квадрат / Правоъгълник
    cv2.rectangle(img, (50, 50), (180, 150), (255, 255, 255), -1)

    # Окръжност
    cv2.circle(img, (350, 100), 50, (255, 255, 255), -1)

    # Наклонена елипса
    cv2.ellipse(img, (150, 300), (70, 35), 45, 0, 360, (255, 255, 255), -1)

    # Триъгълник
    pts = np.array([[350, 220], [280, 350], [420, 350]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    return img


def analyze_objects(image_path=None):
    # 1. Зареждане на изображението или генериране на такова
    if image_path is None:
        img = create_sample_image()
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Изображението '{image_path}' не е намерено. Използва се тестово изображение.")
            img = create_sample_image()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Прагова обработка (Thresholding)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # 3. Откриване на контурите
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"=== Общ брой открити обекти: {len(contours)} ===\n")

    output_img = img.copy()

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 100:
            continue

        # --- 1. Център (x, y) ---
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # --- 2. Aspect Ratio ---
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        # --- 3. Extent ---
        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0

        # --- 4. Solidity ---
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0

        # --- 5. Equivalent Diameter ---
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # --- 6. Orientation ---
        if len(cnt) >= 5:
            (center), (MA, ma), angle = cv2.fitEllipse(cnt)
            orientation = angle
        else:
            orientation = 0.0

        # --- 7. Mean Intensity ---
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]

        # Принтиране в конзолата
        print(f"Обект #{i + 1}:")
        print(f"  - Център (x, y): ({cX}, {cY})")
        print(f"  - Aspect Ratio: {aspect_ratio:.4f}")
        print(f"  - Extent: {extent:.4f}")
        print(f"  - Solidity: {solidity:.4f}")
        print(f"  - Equivalent Diameter: {equivalent_diameter:.2f} px")
        print(f"  - Orientation: {orientation:.2f}°")
        print(f"  - Mean Intensity: {mean_val:.2f}")
        print("-" * 40)

        # Рисуване и номерация
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(output_img, (cX, cY), 4, (0, 0, 255), -1)
        cv2.putText(output_img, f"#{i + 1}", (cX - 15, cY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 4. Визуализация
    cv2.imshow("Binary Threshold", thresh)
    cv2.imshow("Task 4 - Object Analysis", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# === ИЗВИКВАНЕ НА ФУНКЦИЯТА (БЕЗ ДА Е КОМЕНТИРАНО) ===
analyze_objects()ь