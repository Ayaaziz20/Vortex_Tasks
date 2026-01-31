import cv2
import numpy as np

points = []


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_mouse_points(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Original Image", img_display)
            print(f"Points: {x}, {y}")

            if len(points) == 4:
                perform_perspective_transform()


def perform_perspective_transform():
    global points, img
    pts1 = np.float32(points)
    pts1 = order_points(pts1)

    (tl, tr, br, bl) = pts1
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    pts2 = np.float32([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))

    cv2.imshow("Bird's Eye View", result)
    print("DONE!")


img = cv2.imread(r'C:\Users\ayaaz\PycharmProjects\Tasks\Results\Resources\task8\jhonsmith.jpg')
if img is None:
    print("WRONG IMAGE")
else:
    img_display = img.copy()
    cv2.namedWindow("Original Image")
    cv2.setMouseCallback("Original Image", get_mouse_points)

    print("POINTS!!")
    print(" Esc .")

    cv2.imshow("Original Image", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()