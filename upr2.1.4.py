import cv2
import numpy as np

# 1. Текстова решетка (изображение с хоризонтални и вертикални линии)
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), 255, 2)  # Контур на квадрат
cv2.line(img, (50, 150), (250, 150), 255, 2)       # Хоризонтална линия
cv2.line(img, (150, 50), (150, 250), 255, 2)       # Вертикална линия

# 2. Дефиниране на ядра за хоризонтални и вертикални линии
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

# 3. Извличане на линиите чрез Отваряне (Opening)
horizontal_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)

# 4. Визуализация
cv2.imshow("Original Lines", img)
cv2.imshow("1.4 Horizontal Lines", horizontal_lines)
cv2.imshow("1.4 Vertical Lines", vertical_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()