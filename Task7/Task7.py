import cv2
import numpy as np

def detect_and_count_red_targets(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening: {video_path}")
        return

    total_unique_targets = 0
    tracked_centroids = []
    # Distance threshold adjusted for movement between frames
    dist_threshold = 120

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-processing for noise reduction
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Broadened Red range to detect targets in low saturation/underwater conditions
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.add(mask1, mask2)

        # Better morphological operations to close gaps in targets
        kernel = np.ones((15, 15), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Increased area filter to prevent counting noise/reflections
            if area < 1500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cX, cY = x + w // 2, y + h // 2
            current_center = (cX, cY)

            is_new = True
            for i, prev_center in enumerate(tracked_centroids):
                distance = np.sqrt((cX - prev_center[0])**2 + (cY - prev_center[1])**2)

                if distance < dist_threshold:
                    is_new = False
                    tracked_centroids[i] = current_center # Update existing track position
                    break

            if is_new:
                tracked_centroids.append(current_center)
                total_unique_targets += 1

            # Drawing requirements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Counter display
        cv2.rectangle(frame, (5, 5), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Unique Targets: {total_unique_targets}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Task 7 - Detection and Counting", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_files = [
    r'C:\Users\ayaaz\PycharmProjects\Tasks\Results\Resources\Task7\RTSPCamera5.mp4',
    r'C:\Users\ayaaz\PycharmProjects\Tasks\Results\Resources\Task7\RTSPCamera6.mp4',
    r'C:\Users\ayaaz\PycharmProjects\Tasks\Results\Resources\Task7\RTSPCamera7.mp4'
]

for video in video_files:
    detect_and_count_red_targets(video)