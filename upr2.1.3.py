import cv2
import numpy as np

# 1. Текстово бинарно изображение
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (200, 200), 255, -1)

# 2. Дефиниране на ядро за откриване на горен ляв ъгъл
# 1 = бял пиксел, -1 = черен пиксел, 0 = без значение
kernel = np.array([
    [-1, -1, -1],
    [-1,  1,  0],
    [-1,  0,  1]
], dtype=np.int32)

# 3. Прилагане на Hit-or-Miss
hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

# 4. Визуализация
cv2.imshow("Original Image", img)
cv2.imshow("1.3 Hit-or-Miss (Detected Upper-Left Corner)", hit_or_miss)
cv2.waitKey(0)
cv2.destroyAllWindows()