import cv2
import numpy as np

# 1. Създаваме бинарно изображение (черно платно с бял квадрат и шум)
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), 255, -1)

# Добавяме шум (външен бял квадрат и вътрешна черна дупка)
img[20:30, 20:30] = 255
img[120:130, 120:130] = 0

# 2. Дефиниране на ядро (Structuring Element) 5x5
kernel = np.ones((5, 5), np.uint8)

# 3. Ерозия (премахва външни бели пиксели / свива обекта)
erosion = cv2.erode(img, kernel, iterations=1)

# 4. Дилатация (добавя бели пиксели по границите / разширява обекта)
dilation = cv2.dilate(img, kernel, iterations=1)

# 5. Визуализация
cv2.imshow("Original Image", img)
cv2.imshow("1.1 Erosion", erosion)
cv2.imshow("1.1 Dilation", dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()