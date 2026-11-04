import cv2 as cv
import sys

# 1. Зареждане на изображението
image_path = "starry_night.jpg"
src = cv.imread(image_path)

if src is None:
    sys.exit(f"Грешка: Не може да се отвори изображението '{image_path}'.")

# 2. Предварително изглаждане за намаляване на шума (както е в документацията)
# Без лек blur, шумът се интерпретира като фалшиви контури
src_blurred = cv.GaussianBlur(src, (3, 3), 0)

# 3. Преобразуване в степени на сивото (Grayscale)
# Операторът на Собел работи върху едноканални изображения
gray = cv.cvtColor(src_blurred, cv.COLOR_BGR2GRAY)

# Параметри за Sobel
# CV_16S предотвратява препълване при отрицателни градиенти
ddepth = cv.CV_16S
ksize = 3
scale = 1
delta = 0

# 4. Пресмятане на производните по X и Y
# Градиент по X (открива вертикални контури)
grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
# Градиент по Y (открива хоризонтални контури)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

# 5. Преобразуване обратно към 8-битов формат (CV_8U) с абсолютни стойности
abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

# 6. Комбиниране на двата градиента: общ градиент = 0.5 * grad_x + 0.5 * grad_y
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# 7. Визуализиране
cv.imshow("1. Original Grayscale", gray)
cv.imshow("2. Sobel X (Vertical edges)", abs_grad_x)
cv.imshow("3. Sobel Y (Horizontal edges)", abs_grad_y)
cv.imshow("4. Sobel Combined (Full edges)", grad)

# 8. Съхраняване на резултата
cv.imwrite("sobel_combined.png", grad)
print("Изображението със Sobel контури е запазено като 'sobel_combined.png'!")

cv.waitKey(0)
cv.destroyAllWindows()