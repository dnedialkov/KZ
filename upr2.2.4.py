import cv2
import numpy as np

img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_img = img.copy()
for i, cnt in enumerate(contours):
    # Изчисления
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    # Центроид
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    print(f"Контур #{i+1}:")
    print(f"  - Площ (Area): {area}")
    print(f"  - Периметър (Perimeter): {perimeter:.2f}")
    print(f"  - Център (cX, cY): ({cX}, {cY})\n")

    # Визуализация
    cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(output_img, (cX, cY), 5, (0, 0, 255), -1)

cv2.imshow("2.4 Area and Perimeter", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()