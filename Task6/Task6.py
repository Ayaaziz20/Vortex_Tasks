import cv2
import numpy as np
import os


def get_color_name(hsv_pixel):
    h, s, v = hsv_pixel
    # إذا كان اللون باهت جداً (إضاءة عالية أو تشبع منخفض)
    if s < 50: return "White/Gray"

    if (h < 10) or (h > 160):
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 130:
        return "Blue"
    elif 130 <= h < 160:
        return "Purple"
    return "Unknown"


def detect_shape_and_color(full_image_path):
    if not os.path.exists(full_image_path):
        print(f"Error: File not found")
        return

    img = cv2.imread(full_image_path)
    if img is None: return

    output = img.copy()

    # 1. تنعيم قوي جداً للصورة لإخفاء الخطوط الرفيعة والأنابيب والتركيز على الكتل الملونة
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. تحويل الصورة لرمادي ثم عمل Threshold قوي
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. تنظيف الصورة من النقاط الصغيرة (الضوضاء)
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ترتيب الكنتورات حسب المساحة واختيار الأكبر (لأن الشكل المطلوب غالباً هو الأكبر)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000: continue  # تجاهل أي شيء صغير (خطوط، أنابيب، ضوضاء)

        # 4. تحليل الشكل باستخدام تقريب المضلعات
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)

        shape = "Unknown"
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        else:
            # استخدام الدائرية للتفريق بين الدائرة وأي شكل آخر
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity > 0.7:
                shape = "Circle"
            else:
                shape = "Polygon"

        # 5. تحديد اللون
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(hsv, mask=mask)[:3]
        color_name = get_color_name(mean_val)

        # رسم النتيجة
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            text = f"{color_name} {shape}"
            cv2.putText(output, text, (cX - 70, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # نكتفي بأول شكل كبير نكتشفه (لأنه المطلوب في المركز)
        break

    cv2.imshow("Final Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# المسار الصحيح
path_to_image = r'C:\Users\ayaaz\PyCharmProjects\Tasks\Results\Resources\Task6\clean_213601.jpg'
detect_shape_and_color(path_to_image)
