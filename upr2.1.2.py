import cv2
import numpy as np

# 1. Текстово бинарно изображение с шум
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), 255, -1)

# Добавяне на шум:
img[20:30, 20:30] = 255      # Шум извън квадрата (бяла точка)
img[120:130, 120:130] = 0    # Шум вътре в квадрата (черна дупка)

# 2. Дефиниране на ядро (Structuring Element) 5x5
kernel = np.ones((5, 5), np.uint8)

# 3. Прилагане на Opening и Closing
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 4. Визуализация
cv2.imshow("Original with Noise", img)
cv2.imshow("1.2 Opening (Removes outer noise)", opening)
cv2.imshow("1.2 Closing (Fills inner holes)", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()