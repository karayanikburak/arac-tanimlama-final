import cv2
import time

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture('video.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

car_count = 0
cars_detected = []

line_y = frame_height // 2

distance_threshold = 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    new_cars_detected = []
    for (x, y, w, h) in cars:
        center_x = x + w // 2
        center_y = y + h // 2

        is_new_car = True
        for (prev_x, prev_y) in cars_detected:
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
            if distance < distance_threshold:
                is_new_car = False
                break
        
        if is_new_car and center_y > line_y:
            car_count += 1
            new_cars_detected.append((center_x, center_y))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cars_detected.extend(new_cars_detected)
    
    cv2.putText(frame, f"ARACLAR: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)
    
    cv2.imshow('ARAC TANIMLAMA', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()