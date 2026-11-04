import cv2 as cv
import sys

# 1. Зареждане на изображението
image_path = "starry_night.jpg"
src = cv.imread(image_path)

if src is None:
    sys.exit(f"Грешка: Не може да се отвори изображението '{image_path}'.")

# 2. Намаляване на шума с GaussianBlur (ядро 3x3)
src_blurred = cv.GaussianBlur(src, (3, 3), 0)

# 3. Преобразуване в степени на сивото
gray = cv.cvtColor(src_blurred, cv.COLOR_BGR2GRAY)

# Параметри за Лапласиана
# CV_16S предотвратява препълване при отрицателни втори производни
ddepth = cv.CV_16S
kernel_size = 3

# 4. Прилагане на оператора на Лаплас (втора производна)
dst = cv.Laplacian(gray, ddepth, ksize=kernel_size)

# 5. Преобразуване обратно към 8-битов масив (CV_8U) с абсолютни стойности
abs_dst = cv.convertScaleAbs(dst)

# 6. Визуализиране
cv.imshow("1. Grayscale + Blur", gray)
cv.imshow("2. Laplace Edge Detector", abs_dst)

# 7. Съхраняване на резултата
cv.imwrite("laplace_edges.png", abs_dst)
print("Изображението с контури по Лаплас е запазено като 'laplace_edges.png'!")

cv.waitKey(0)
cv.destroyAllWindows()