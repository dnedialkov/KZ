import cv2 as cv
import sys

# 1. Зареждане на оригиналното изображение
image_path = "starry_night.jpg"
src = cv.imread(image_path)

if src is None:
    sys.exit(f"Грешка: Не може да се отвори изображението '{image_path}'.")

# Размер на ядрото (трябва да е нечетно положително число: 3, 5, 7, 9, 11...)
ksize = 11

# 2. Прилагане на 4-те филтъра от документацията:

# А) Обикновен средноаритметичен филтър (Homogeneous blur / Box filter)
dst_homogeneous = cv.blur(src, (ksize, ksize))

# Б) Гаусов филтър (Gaussian Blur)
# (ksize, ksize) е размерът на ядрото, 0 означава, че sigma се изчислява автоматично
dst_gaussian = cv.GaussianBlur(src, (ksize, ksize), 0)

# В) Медианен филтър (Median Blur)
# Тук размерът на ядрото се подава като едно число (int)
dst_median = cv.medianBlur(src, ksize)

# Г) Билатерален филтър (Bilateral Filter)
# d: диаметър на съседството (напр. 9)
# sigmaColor: филтър в цветовото пространство (напр. 75)
# sigmaSpace: филтър в координатното пространство (напр. 75)
dst_bilateral = cv.bilateralFilter(src, d=9, sigmaColor=75, sigmaSpace=75)

# 3. Визуализиране на резултатите в отделни прозорци
cv.imshow("1. Original", src)
cv.imshow("2. Homogeneous Blur", dst_homogeneous)
cv.imshow("3. Gaussian Blur", dst_gaussian)
cv.imshow("4. Median Blur", dst_median)
cv.imshow("5. Bilateral Filter", dst_bilateral)

# 4. Съхраняване на резултатите
cv.imwrite("blur_homogeneous.png", dst_homogeneous)
cv.imwrite("blur_gaussian.png", dst_gaussian)
cv.imwrite("blur_median.png", dst_median)
cv.imwrite("blur_bilateral.png", dst_bilateral)
print("Филтрираните изображения са запазени успешно като .png файлове!")

# Изчакване на натискане на клавиш за затваряне
cv.waitKey(0)
cv.destroyAllWindows()