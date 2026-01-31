import cv2
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def smart_resize_to_width(img, target_width):
    h, w = img.shape[:2]
    aspect_ratio = h / w
    return cv2.resize(img, (target_width, int(target_width * aspect_ratio)))


def combine_stream_horizontally(images, name):
    if not images: return None

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, stitched = stitcher.stitch(images)

    if status == cv2.STITCHER_OK:
        print(f"✅ {name}: !")
        return stitched
    else:
        print(f"⚠️ {name}: Done")
        h_min = min(img.shape[0] for img in images)
        resized = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min)) for img in images]
        return cv2.hconcat(resized)


def create_mosaic():
    base_path = "../Resources/Task5/"

    stream1_imgs = []
    stream2_imgs = []

    print("--- 📥 Loading Images ---")
    for i in range(1, 6):
        p1 = os.path.join(base_path, "Stream 1", f"s{i}.jpg")
        p2 = os.path.join(base_path, "Stream 2", f"s{i}.png")

        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)

        if img1 is not None: stream1_imgs.append(img1)
        if img2 is not None: stream2_imgs.append(img2)

    if not stream1_imgs or not stream2_imgs:
        print("❌ Wrong")
        return

    row1 = combine_stream_horizontally(stream1_imgs, "Stream 1")
    row2 = combine_stream_horizontally(stream2_imgs, "Stream 2")

    row2_final = cv2.resize(row2, (row1.shape[1], int(row2.shape[0] * row1.shape[1] / row2.shape[1])))

    final_mosaic = cv2.vconcat([row1, row2_final])

    cv2.imwrite("final_reef_mosaic.jpg", final_mosaic)

    screen_w = 1200
    display_h = int(final_mosaic.shape[0] * (screen_w / final_mosaic.shape[1]))
    cv2.imshow("Autonomous Photomosaic Result", cv2.resize(final_mosaic, (screen_w, display_h)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    create_mosaic()