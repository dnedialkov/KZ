import cv2 as cv
import sys

# 1. Зареждане директно в степени на сивото
image_path = "starry_night.jpg"
src = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if src is None:
    sys.exit(f"Грешка: Не може да се отвори изображението '{image_path}'.")

window_name = "6. Threshold Demo"
cv.namedWindow(window_name)

# Начални стойности за плъзгачите
threshold_value = 127
max_value = 255
threshold_type = 0  # 0: Binary, 1: Binary Inverted, 2: Truncate, 3: To Zero, 4: To Zero Inverted

trackbar_type = "Type: 0-Bin, 1-Inv, 2-Trunc, 3-Zero, 4-ZeroInv"
trackbar_value = "Value"

def update_threshold(_=None):
    """Callback функция, която пресмята новия праг при промяна на плъзгачите."""
    t_type = cv.getTrackbarPos(trackbar_type, window_name)
    t_val = cv.getTrackbarPos(trackbar_value, window_name)

    # Прилагане на праговата операция
    _, dst = cv.threshold(src, t_val, max_value, t_type)

    cv.imshow(window_name, dst)
    cv.imwrite("threshold_output.png", dst)

# Създаване на плъзгачи за избор на тип (0-4) и стойност на прага (0-255)
cv.createTrackbar(trackbar_type, window_name, threshold_type, 4, update_threshold)
cv.createTrackbar(trackbar_value, window_name, threshold_value, 255, update_threshold)

# Първоначално визуализиране
update_threshold()

print("Променяйте Type (0 до 4) и Value (0 до 255) с мишката.")
print("Натиснете произволен клавиш за изход...")

cv.waitKey(0)
cv.destroyAllWindows()