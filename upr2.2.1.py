import cv2
import numpy as np

# 1. Текстово бинарно изображение с няколко форми
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (120, 120), (255, 255, 255), -1)
cv2.circle(img, (200, 200), 40, (255, 255, 255), -1)

# 2. Сивостепенно изображение и Thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 3. Откриване на контури
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 4. Рисуване на контурите (в зелено)
output_img = img.copy()
cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)

cv2.imshow("2.1 Find Contours", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()