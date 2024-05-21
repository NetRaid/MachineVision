import cv2
import numpy as np

# def capture_images_from_camera(camera_index=0, save_directory="captured_frames", display_window=True):
#     # Открыть подключение к камере
#     cap = cv2.VideoCapture(camera_index)
    
#     if not cap.isOpened():
#         print("Ошибка: не удалось подключиться к камере.")
#         return
    
#     # Создаем директорию для сохранения изображений, если ее нет
#     import os
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
    
#     frame_count = 0
    
#     while True:
#         # Захват кадра
#         ret, frame = cap.read()
        
#         if not ret:
#             print("Ошибка: не удалось захватить кадр.")
#             break
        
#         # Отображение кадра
#         if display_window:
#             cv2.imshow('Camera Feed', frame)
        
#         # Сохранение кадра
#         frame_filename = os.path.join(save_directory, f"frame_{frame_count:04d}.png")
#         cv2.imwrite(frame_filename, frame)
#         frame_count += 1
        
#         # Выход из цикла по нажатию клавиши 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Освобождение ресурсов
#     cap.release()
#     if display_window:
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     capture_images_from_camera()



def filter_noise(image):
    """
    Фильтрация шума с использованием гауссовского размытия.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Коррекция яркости и контрастности изображения.
    Параметры:
        brightness: значение яркости от -100 до 100.
        contrast: значение контрастности от -100 до 100.
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    return buf

def enhance_image(image):
    """
    Улучшение качества изображения.
    """
    # Применяем CLAHE (Контрастное Ограничение Адаптивной Гистограммы Эквализации)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def process_image(image_path, output_path):
    """
    Полный процесс обработки изображения: фильтрация шума, коррекция яркости и контрастности, улучшение качества.
    """
    # Загружаем изображение
    image = cv2.imread("test2.jpg")

    # Фильтрация шума
    noise_filtered = filter_noise(image)

    # Коррекция яркости и контрастности
    brightness_contrast_adjusted = adjust_brightness_contrast(noise_filtered, brightness=10, contrast=30)

    # Улучшение качества изображения
    enhanced_image = enhance_image(brightness_contrast_adjusted)

    # Сохранение обработанного изображения
    # cv2.imwrite(output_path, enhanced_image)
    cv2.imshow("gg", enhanced_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    # Пример использования
    input_image_path = "input.jpg"
    output_image_path = "output.jpg"
    process_image(input_image_path, output_image_path)