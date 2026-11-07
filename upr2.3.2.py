import cv2
import numpy as np

# 1. Генериране на тестово изображение с окръжности
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
cv2.circle(img, (250, 250), 65, (255, 255, 255), -1)
cv2.circle(img, (300, 100), 25, (255, 255, 255), -1)

# 2. Преобразуване в сивостепенно и медианно размиване за премахване на шум
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.medianBlur(gray, 5)

# 3. Прилагане на HoughCircles
circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=50,
    param2=15,
    minRadius=10,
    maxRadius=100
)

# 4. Изчертаване на откритите окръжности
output_img = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Външна окръжност (зелена)
        cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Център (червена точка)
        cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow("3.2 Detected Circles (HoughCircles)", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()