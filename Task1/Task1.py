import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mode = 'original'
rotate_count = 0
recording = False
video_writer = None

print("Keys: Q=Quit, R=Rotate, C=Capture Image, S=Record Video, G=Gray, H=HSV, X=All, Z=Original")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    for _ in range(rotate_count % 4):
        display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mode == 'gray':
        final_output = gray_frame
    elif mode == 'hsv':
        final_output = hsv_frame
    elif mode == 'all':

        top_row = np.hstack((frame, gray_bgr))

        rotated_for_all = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rotated_res = cv2.resize(rotated_for_all, (frame.shape[1], frame.shape[0]))
        bottom_row = np.hstack((hsv_frame, rotated_res))
        final_output = np.vstack((top_row, bottom_row))
    else:
        final_output = display_frame

    cv2.imshow('Camera App', final_output)

    if recording and video_writer is not None:
        video_writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        rotate_count += 1
    elif key == ord('c'):
        cv2.imwrite('captured_image.jpg', display_frame)
        print("Image Saved!")
    elif key == ord('s'):
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            print("Recording Started...")
        else:
            recording = False
            video_writer.release()
            print("Recording Stopped & Saved!")
    elif key == ord('g'):
        mode = 'gray'
    elif key == ord('h'):
        mode = 'hsv'
    elif key == ord('x'):
        mode = 'all'
    elif key == ord('z'):
        mode = 'original'

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()