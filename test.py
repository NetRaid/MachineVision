import cv2
import numpy as np

def adjust_brightness_contrast(image, brightness=40, contrast=1.3):
    beta = brightness
    alpha = contrast / 127 + 1
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def find_contours_and_centers(image, lower_bound, upper_bound, min_area=2300):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    centers = []

    for contour in large_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            centers.append(None)
    return large_contours, centers

def get_center_between_two_contours(center1, center2):
    if center1 and center2:
        center_between = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        return center_between
    return None

def process_frame(frame):
    # Фильтрация шума
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Коррекция яркости и контрастности
    frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)

    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение цветовых диапазонов
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    yellow_lower = np.array([22, 75, 75])
    yellow_upper = np.array([35, 255, 255])

    blue_lower = np.array([100, 150, 100])
    blue_upper = np.array([140, 255, 255])

    # Обнаружение контуров и центров для каждого цвета
    green_contours, green_centers = find_contours_and_centers(hsv, green_lower, green_upper)
    yellow_contours, yellow_centers = find_contours_and_centers(hsv, yellow_lower, yellow_upper)
    blue_contours, blue_centers = find_contours_and_centers(hsv, blue_lower, blue_upper)

    # Обработка контуров и центров
    for contours, centers, color in zip(
        [green_contours, yellow_contours, blue_contours],
        [green_centers, yellow_centers, blue_centers],
        [(0, 255, 0), (0, 255, 255), (255, 0, 0)]
    ):
        for contour, center in zip(contours, centers):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if center:
                cv2.circle(frame, center, 5, color, -1)
                cv2.putText(frame, f"Center: {center}", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # cv2.putText(frame, f"Top-left: ({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Нахождение центра между двумя большими синими контурами
    if len(yellow_centers) >= 2:
        center_between = get_center_between_two_contours(yellow_centers[0], yellow_centers[1])
        if center_between:
            cv2.circle(frame, center_between, 5, (255, 255, 255), -1)
            cv2.putText(frame, f"Center between: {center_between}", (center_between[0] + 10, center_between[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к камере.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр.")
            break

        processed_frame = process_frame(frame)
        
        cv2.imshow('Processed Video', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()