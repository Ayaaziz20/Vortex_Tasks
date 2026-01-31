import cv2
import numpy as np
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_coral_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_pink = np.array([140, 40, 40])
    upper_pink = np.array([180, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    lower_white = np.array([0, 0, 165])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

    return mask_pink, mask_white


base_path = "../Resources/Task3/"

ref_img = cv2.imread(base_path + 'OneYearImage.jpg')

if ref_img is None:
    print(f"❌ NOT FOUND {base_path}OneYearImage.jpg")
    print(" Resources Not in Task3")
    exit()

ref_img = cv2.resize(ref_img, (700, 500))
ref_pink, ref_white = get_coral_masks(ref_img)
ref_total = cv2.bitwise_or(ref_pink, ref_white)

images_to_check = ['coral1.jpg', 'coral2.jpg', 'coral3.jpg', 'coral4.jpg', 'coral5.jpg', 'coral6.jpeg']

for img_name in images_to_check:
    full_path = base_path + img_name
    current_img = cv2.imread(full_path)

    if current_img is None:
        print(f"⚠️  {full_path}")
        continue

    current_img = cv2.resize(current_img, (700, 500))
    cur_pink, cur_white = get_coral_masks(current_img)
    cur_total = cv2.bitwise_or(cur_pink, cur_white)


    kernel_dil = np.ones((7, 7), np.uint8)
    ref_total_dil = cv2.dilate(ref_total, kernel_dil)
    ref_pink_dil = cv2.dilate(ref_pink, kernel_dil)
    ref_white_dil = cv2.dilate(ref_white, kernel_dil)

    death = cv2.subtract(ref_total_dil, cur_total)

    bleaching = cv2.bitwise_and(ref_pink_dil, cur_white)

    recovery = cv2.bitwise_and(ref_white_dil, cur_pink)


    def draw(img, mask, color, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


    draw(current_img, death, (0, 255, 255), "Dead")
    draw(current_img, bleaching, (0, 0, 255), "Bleached")
    draw(current_img, recovery, (255, 0, 0), "Recovered")  

    cv2.imshow(f"Analysis: {img_name}", current_img)
    print(f"✅ Done {img_name}")

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()