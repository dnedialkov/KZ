import cv2
import numpy as np

# 1. Текстово изображение с наклонена елипса/форма
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.ellipse(img, (200, 150), (80, 40), 30, 0, 360, (255, 255, 255), -1)

# 2. Преобразуване в сивостепенно и прагова обработка (Threshold)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 3. Откриване на контурите
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_img = img.copy()

for cnt in contours:
    # fitEllipse изисква минимум 5 точки в контура
    if len(cnt) >= 5:
        # --- 1. Ротиран правоъгълник (Жълт) ---
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Преобразуване в цели числа
        cv2.drawContours(output_img, [box], 0, (0, 255, 255), 2)

        # --- 2. Ограждаща елипса (Синя/Циан) ---
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(output_img, ellipse, (255, 255, 0), 2)

# 4. Визуализация
cv2.imshow("2.3 Rotated Rect & Fit Ellipse", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()