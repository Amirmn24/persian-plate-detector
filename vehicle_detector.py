"""
ماژول تشخیص خودرو در تصاویر
با استفاده از مدل YOLOv8 از کتابخانه ultralytics
"""

import cv2
import os
import sys
from ultralytics import YOLO


# کلاس‌های مربوط به وسایل نقلیه در مدل COCO (که YOLOv8 روی آن آموزش دیده)
VEHICLE_CLASSES = {
    2: "car",        # ماشین سواری
    3: "motorcycle",  # موتورسیکلت
    5: "bus",         # اتوبوس
    7: "truck",       # کامیون
}


class VehicleDetector:
    """کلاس اصلی برای تشخیص خودرو در تصاویر"""

    def __init__(self, model_name="yolov8n.pt", confidence=0.5):
        """
        مقداردهی اولیه

        Args:
            model_name: نام مدل YOLOv8 (پیش‌فرض: yolov8n.pt - سبک‌ترین مدل)
            confidence: حداقل میزان اطمینان برای تشخیص (0 تا 1)
        """
        print(f"[*] در حال بارگذاری مدل {model_name} ...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        print("[+] مدل با موفقیت بارگذاری شد!")

    def detect_vehicles(self, image_path):
        """
        تشخیص خودرو در یک تصویر

        Args:
            image_path: مسیر فایل تصویر

        Returns:
            dict: نتیجه تشخیص شامل وجود خودرو، تعداد و جزئیات
        """
        # بررسی وجود فایل
        if not os.path.exists(image_path):
            print(f"[!] خطا: فایل '{image_path}' یافت نشد.")
            return None

        # خواندن تصویر
        image = cv2.imread(image_path)
        if image is None:
            print(f"[!] خطا: نمی‌توان تصویر '{image_path}' را خواند.")
            return None

        # اجرای مدل روی تصویر
        results = self.model(image, conf=self.confidence, verbose=False)

        # استخراج وسایل نقلیه از نتایج
        vehicles_found = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id in VEHICLE_CLASSES:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicles_found.append({
                        "type": VEHICLE_CLASSES[class_id],
                        "confidence": round(conf, 2),
                        "bbox": (x1, y1, x2, y2),
                    })

        return {
            "image_path": image_path,
            "has_vehicle": len(vehicles_found) > 0,
            "vehicle_count": len(vehicles_found),
            "vehicles": vehicles_found,
        }

    def detect_and_draw(self, image_path, output_path=None):
        """
        تشخیص خودرو و رسم کادر دور آن‌ها

        Args:
            image_path: مسیر فایل تصویر ورودی
            output_path: مسیر ذخیره تصویر خروجی (اختیاری)

        Returns:
            dict: نتیجه تشخیص
        """
        detection = self.detect_vehicles(image_path)
        if detection is None:
            return None

        image = cv2.imread(image_path)

        # رنگ‌ها برای انواع مختلف وسایل نقلیه
        colors = {
            "car": (0, 255, 0),        # سبز
            "motorcycle": (255, 0, 0),   # آبی
            "bus": (0, 165, 255),        # نارنجی
            "truck": (0, 0, 255),        # قرمز
        }

        for vehicle in detection["vehicles"]:
            x1, y1, x2, y2 = vehicle["bbox"]
            color = colors.get(vehicle["type"], (255, 255, 255))
            label = f"{vehicle['type']} ({vehicle['confidence']:.0%})"

            # رسم کادر
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # رسم برچسب
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        # ذخیره یا نمایش تصویر
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"[+] تصویر خروجی ذخیره شد: {output_path}")
        else:
            cv2.imshow("Vehicle Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detection

    def process_folder(self, folder_path, output_folder=None):
        """
        پردازش تمام تصاویر یک پوشه

        Args:
            folder_path: مسیر پوشه حاوی تصاویر
            output_folder: مسیر پوشه خروجی (اختیاری)

        Returns:
            list: لیست نتایج تشخیص
        """
        if not os.path.isdir(folder_path):
            print(f"[!] خطا: پوشه '{folder_path}' یافت نشد.")
            return []

        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_extensions)
        ]

        if not image_files:
            print("[!] هیچ تصویری در پوشه یافت نشد.")
            return []

        print(f"[*] تعداد {len(image_files)} تصویر یافت شد. شروع پردازش...\n")

        all_results = []
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, filename)
            print(f"--- تصویر {i}/{len(image_files)}: {filename} ---")

            if output_folder:
                out_path = os.path.join(output_folder, f"detected_{filename}")
                result = self.detect_and_draw(image_path, out_path)
            else:
                result = self.detect_vehicles(image_path)

            if result:
                all_results.append(result)
                _print_result(result)

        # نمایش خلاصه
        print("\n" + "=" * 50)
        print("خلاصه نتایج:")
        print(f"  کل تصاویر پردازش شده: {len(all_results)}")
        with_vehicle = sum(1 for r in all_results if r["has_vehicle"])
        print(f"  تصاویر دارای خودرو:   {with_vehicle}")
        print(f"  تصاویر بدون خودرو:    {len(all_results) - with_vehicle}")
        print("=" * 50)

        return all_results


def _print_result(result):
    """چاپ نتیجه تشخیص به صورت خوانا"""
    if result["has_vehicle"]:
        print(f"  [+] خودرو شناسایی شد! تعداد: {result['vehicle_count']}")
        for v in result["vehicles"]:
            print(f"      - {v['type']} (اطمینان: {v['confidence']:.0%})")
    else:
        print("  [-] هیچ خودرویی شناسایی نشد.")
    print()


def main():
    """تابع اصلی برنامه"""
    print("=" * 50)
    print("   سیستم تشخیص خودرو در تصاویر")
    print("   Vehicle Detection System")
    print("=" * 50)
    print()

    # ساخت شیء تشخیص‌دهنده
    detector = VehicleDetector(model_name="yolov8n.pt", confidence=0.5)
    print()

    if len(sys.argv) < 2:
        print("نحوه استفاده:")
        print("  تشخیص در یک تصویر:")
        print("    python vehicle_detector.py image.jpg")
        print()
        print("  تشخیص در یک پوشه از تصاویر:")
        print("    python vehicle_detector.py images_folder/")
        print()
        print("  تشخیص و ذخیره خروجی:")
        print("    python vehicle_detector.py images_folder/ --output results/")
        return

    input_path = sys.argv[1]
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    # اگر مسیر یک فایل باشد
    if os.path.isfile(input_path):
        if output_path:
            result = detector.detect_and_draw(input_path, output_path)
        else:
            result = detector.detect_and_draw(input_path)
        if result:
            _print_result(result)

    # اگر مسیر یک پوشه باشد
    elif os.path.isdir(input_path):
        detector.process_folder(input_path, output_path)

    else:
        print(f"[!] خطا: مسیر '{input_path}' معتبر نیست.")


if __name__ == "__main__":
    main()
