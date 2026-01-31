import cv2
import numpy as np

img = cv2.imread(r'C:\Users\ayaaz\PycharmProjects\Tasks\Results\Resources\task8\computer vision.PNG')

src_pts = np.float32([
    [279, 181],
    [481, 292],
    [276, 564],
    [48, 365]
])

width, height = 500, 700

dst_pts = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Original", img)
cv2.imshow("Perspective Transform", result)
cv2.waitKey(0)
cv2.destroyAllWindows()