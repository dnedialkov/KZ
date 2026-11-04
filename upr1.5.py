import cv2 as cv
import sys

# 1. Зареждане на изображението директно в степени на сивото
image_path = "starry_night.jpg"
src_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if src_gray is None:
    sys.exit(f"Грешка: Не може да се отвори изображението '{image_path}'.")

# 2. Предварително леко замъгляване (3x3), за да няма фалшиви контури от шум
src_gray = cv.blur(src_gray, (3, 3))

window_name = "5. Canny Edge Detector"
cv.namedWindow(window_name)

# Начални стойности за праговете
low_threshold = 50
max_lowThreshold = 100
ratio = 3
kernel_size = 3

def canny_threshold(val):
    """Callback функция, която се вика при местене на плъзгача."""
    # Горният праг се задава като ratio * low_threshold (препоръчително 2:1 или 3:1)
    detected_edges = cv.Canny(src_gray, val, val * ratio, apertureSize=kernel_size)
    cv.imshow(window_name, detected_edges)
    # Записва текущия резултат
    cv.imwrite("canny_edges.png", detected_edges)

# Създаване на интерактивен Trackbar (плъзгач) в прозореца
cv.createTrackbar("Min Threshold:", window_name, low_threshold, max_lowThreshold, canny_threshold)

# Първоначално извикване за показване на старта
canny_threshold(low_threshold)
print("Движете плъзгача 'Min Threshold:' с мишката, за да променяте чувствителността на Canny.")
print("Натиснете произволен клавиш за изход...")

cv.waitKey(0)
cv.destroyAllWindows()