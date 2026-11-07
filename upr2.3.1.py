import cv2
import numpy as np

# 1. Генериране на тестово изображение с линии
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.line(img, (50, 50), (350, 50), (255, 255, 255), 2)
cv2.line(img, (50, 50), (50, 350), (255, 255, 255), 2)
cv2.line(img, (50, 350), (350, 350), (255, 255, 255), 2)
cv2.line(img, (100, 100), (300, 300), (255, 255, 255), 2)

# 2. Подготовка: Сивостепенно изображение и детектор на контури Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 3. Прилагане на Вероятностна трансформация на Хаф
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# 4. Визуализация на откритите линии (с безопасно разопаковане)
output_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.ravel()  # ravel() сплесква масива независимо от формата му
        cv2.line(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

cv2.imshow("3.1 Canny Edges", edges)
cv2.imshow("3.1 Detected Lines (HoughLinesP)", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()