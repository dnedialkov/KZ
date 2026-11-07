import cv2
import numpy as np

# 1. Създаваме тестово бинарно изображение (черно платно с бял квадрат и шум)
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), 255, -1)

# Добавяме шум (точки)
img[20:30, 20:30] = 255  # Шум извън квадрата
img[120:130, 120:130] = 0  # Дупка вътре в квадрата

# 2. Дефиниране на структуриращ елемент (ядро 5x5)
kernel = np.ones((5, 5), np.uint8)
# Алтернативно с OpenCV:
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 3. Прилагане на операциите
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 4. Визуализация
cv2.imshow("Original with Noise", img)
cv2.imshow("1.1 Erosion", erosion)
cv2.imshow("1.1 Dilation", dilation)
cv2.imshow("1.2 Opening (removes outer noise)", opening)
cv2.imshow("1.2 Closing (fills inner holes)", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()