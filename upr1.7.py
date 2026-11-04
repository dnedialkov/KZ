import cv2 as cv
import numpy as np
import sys

# 1. Зареждане на входното изображение
image_path = "starry_night.jpg"
src = cv.imread(image_path)

if src is None:
    sys.exit(f"Грешка: Не може да се отвори изображението '{image_path}'.")

# Преобразуване в grayscale (нужно за диференциалните оператори)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# ЕТАП 1: Изглаждащ филтър (Gaussian Blur)
# Намалява високочестотния шум, който би създал грешни градиенти
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# ЕТАП 2: Диференциален оператор (Собел градиент)
# Пресмятане на първите производни по X и Y
sobel_x = cv.Sobel(blurred, cv.CV_16S, 1, 0, ksize=3)
sobel_y = cv.Sobel(blurred, cv.CV_16S, 0, 1, ksize=3)

# Взимане на абсолютни стойности и превръщане обратно в 8-битов масив (0-255)
abs_grad_x = cv.convertScaleAbs(sobel_x)
abs_grad_y = cv.convertScaleAbs(sobel_y)

# Обединяване на хоризонталните и вертикалните очертания
edges_gradient = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# ЕТАП 3: Прагова операция (Бинаризация чрез алгоритъма на Otsu)
# Otsu автоматично изчислява най-добрия праг спрямо хистограмата на градиента
otsu_threshold, binary_result = cv.threshold(
    edges_gradient, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)

print(f"Автоматично изчислен оптимален праг (Otsu): {otsu_threshold}")

# 4. Визуализиране на междинните стъпки и крайния бинарен резултат
cv.imshow("1. Original", src)
cv.imshow("2. Filtered (Gaussian Blur)", blurred)
cv.imshow("3. Differential Operator (Sobel)", edges_gradient)
cv.imshow("4. Final Binary Image (Threshold)", binary_result)

# 5. Съхраняване на финалното бинарно изображениеgit init
output_filename = "task7_binary_output.png"
cv.imwrite(output_filename, binary_result)
print(f"Бинарното изображение е запазено успешно като '{output_filename}'.")

cv.waitKey(0)
cv.destroyAllWindows()