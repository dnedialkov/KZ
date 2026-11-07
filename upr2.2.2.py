import cv2
import numpy as np

img = np.zeros((300, 400, 3), dtype=np.uint8)
# Рисуваме триъгълник за демонстрация
pts = np.array([[100, 50], [50, 200], [200, 200]], np.int32)
cv2.fillPoly(img, [pts], (255, 255, 255))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_img = img.copy()
for cnt in contours:
    # 1. Прав ограждащ правоъгълник (червен)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 2. Ограждаща окръжност (синя)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    center = (int(cx), int(cy))
    radius = int(radius)
    cv2.circle(output_img, center, radius, (255, 0, 0), 2)

cv2.imshow("2.2 Bounding Rect & Circle", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()