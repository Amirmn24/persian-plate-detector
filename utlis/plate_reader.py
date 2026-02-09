"""
ماژول تشخیص و خواندن پلاک ایرانی
با استفاده از YOLO برای تشخیص پلاک و EasyOCR برای خواندن متن
"""

import cv2
import numpy as np
import re
from ultralytics import YOLO


class IranianPlateReader:
    """کلاس اصلی برای تشخیص و خواندن پلاک ایرانی"""
    
    # حروف فارسی مجاز در پلاک ایرانی
    PERSIAN_LETTERS = {
        'الف': 'الف', 'ب': 'ب', 'پ': 'پ', 'ت': 'ت', 'ث': 'ث',
        'ج': 'ج', 'چ': 'چ', 'ح': 'ح', 'خ': 'خ', 'د': 'د',
        'ذ': 'ذ', 'ر': 'ر', 'ز': 'ز', 'ژ': 'ژ', 'س': 'س',
        'ش': 'ش', 'ص': 'ص', 'ض': 'ض', 'ط': 'ط', 'ظ': 'ظ',
        'ع': 'ع', 'غ': 'غ', 'ف': 'ف', 'ق': 'ق', 'ک': 'ک',
        'گ': 'گ', 'ل': 'ل', 'م': 'م', 'ن': 'ن', 'و': 'و',
        'ه': 'ه', 'ی': 'ی',
    }
    
    def __init__(self, plate_model_path="yolov8n.pt", confidence=0.3):
        """
        مقداردهی اولیه
        
        Args:
            plate_model_path: مسیر مدل YOLO برای تشخیص پلاک
            confidence: حداقل میزان اطمینان برای تشخیص
        """
        print(f"[*] در حال بارگذاری مدل تشخیص پلاک...")
        self.plate_model = YOLO(plate_model_path)
        self.confidence = confidence
        self.ocr_reader = None  # به صورت lazy load
        print("[+] مدل تشخیص پلاک آماده است!")
    
    def _get_ocr_reader(self):
        """دریافت یا ایجاد OCR reader (lazy loading)"""
        if self.ocr_reader is None:
            try:
                import easyocr
                print("[*] در حال بارگذاری مدل OCR فارسی...")
                self.ocr_reader = easyocr.Reader(['fa', 'en'], gpu=False)
                print("[+] مدل OCR آماده است!")
            except ImportError:
                print("[!] خطا: کتابخانه easyocr نصب نیست.")
                print("    لطفا با دستور زیر نصب کنید:")
                print("    pip install easyocr")
                return None
        return self.ocr_reader
    
    def detect_plate_region(self, vehicle_image, vehicle_bbox=None):
        """
        تشخیص ناحیه پلاک در تصویر خودرو
        
        Args:
            vehicle_image: تصویر خودرو (numpy array)
            vehicle_bbox: مختصات خودرو در تصویر اصلی (اختیاری)
        
        Returns:
            list: لیست نواحی پلاک شناسایی شده
        """
        # روش 1: استفاده از YOLO برای تشخیص پلاک
        # اگر مدل مخصوص پلاک دارید، از آن استفاده کنید
        # در غیر این صورت از روش پردازش تصویر استفاده می‌کنیم
        
        # روش 2: پردازش تصویر برای یافتن نواحی مستطیلی سفید
        plates = self._detect_plate_by_color(vehicle_image)
        
        return plates
    
    def _detect_plate_by_color(self, image):
        """
        تشخیص پلاک بر اساس رنگ و شکل
        
        Args:
            image: تصویر ورودی
        
        Returns:
            list: لیست نواحی پلاک
        """
        plates = []
        
        # تبدیل به HSV برای تشخیص بهتر رنگ
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # ماسک برای رنگ سفید (پلاک سفید)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # ماسک برای رنگ زرد (پلاک زرد)
        lower_yellow = np.array([15, 80, 150])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # ترکیب ماسک‌ها
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # پردازش مورفولوژی
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # یافتن کانتورها
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # فیلتر کردن کانتورها
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # حداقل اندازه
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # نسبت ابعاد پلاک ایرانی تقریباً 3:1 تا 4:1
            if 2.5 <= aspect_ratio <= 5.0 and w > 80 and h > 20:
                plates.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.7,  # اطمینان تخمینی
                    'area': area
                })
        
        # مرتب‌سازی بر اساس اندازه (بزرگترین اول)
        plates.sort(key=lambda x: x['area'], reverse=True)
        
        return plates
    
    def read_plate_text(self, plate_image):
        """
        خواندن متن پلاک با استفاده از OCR
        
        Args:
            plate_image: تصویر ناحیه پلاک
        
        Returns:
            str: متن پلاک یا None
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return None
        
        # پیش‌پردازش تصویر برای بهبود OCR
        processed = self._preprocess_plate_image(plate_image)
        
        try:
            # خواندن متن
            results = reader.readtext(processed)
            
            if not results:
                return None
            
            # استخراج متن از نتایج
            texts = [text for (bbox, text, confidence) in results if confidence > 0.3]
            full_text = ' '.join(texts)
            
            return full_text
            
        except Exception as e:
            print(f"[!] خطا در OCR: {e}")
            return None
    
    def _preprocess_plate_image(self, image):
        """
        پیش‌پردازش تصویر پلاک برای بهبود OCR
        
        Args:
            image: تصویر پلاک
        
        Returns:
            تصویر پردازش شده
        """
        # تبدیل به grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # افزایش اندازه برای OCR بهتر
        scale_factor = 2
        height, width = gray.shape
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor))
        
        # اعمال threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # کاهش نویز
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def parse_iranian_plate(self, text):
        """
        تجزیه و تحلیل متن پلاک ایرانی
        فرمت: 2 رقم + حرف فارسی + 3 رقم + 2 رقم
        
        Args:
            text: متن خوانده شده از پلاک
        
        Returns:
            dict: اجزای پلاک یا None
        """
        if not text:
            return None
        
        # حذف فاصله‌های اضافی
        text = text.strip()
        
        # الگوهای مختلف برای پلاک ایرانی
        # الگو 1: 12 الف 345 67
        pattern1 = r'(\d{2})\s*([آ-ی])\s*(\d{3})\s*(\d{2})'
        
        # جستجو با regex
        match = re.search(pattern1, text)
        
        if match:
            return {
                'full_text': text,
                'part1': match.group(1),  # 2 رقم اول
                'letter': match.group(2),  # حرف
                'part2': match.group(3),  # 3 رقم وسط
                'part3': match.group(4),  # 2 رقم آخر
                'formatted': f"{match.group(1)} {match.group(2)} {match.group(3)} - {match.group(4)}"
            }
        
        # اگر الگو مطابقت نداشت، سعی می‌کنیم اجزا را جدا کنیم
        return self._parse_flexible(text)
    
    def _parse_flexible(self, text):
        """
        تجزیه انعطاف‌پذیر متن پلاک
        
        Args:
            text: متن پلاک
        
        Returns:
            dict یا None
        """
        # استخراج ارقام و حروف
        digits = re.findall(r'\d+', text)
        letters = re.findall(r'[آ-ی]', text)
        
        # بررسی تعداد ارقام و حروف
        all_digits = ''.join(digits)
        
        if len(all_digits) >= 7 and len(letters) >= 1:
            # تلاش برای تطبیق با فرمت
            try:
                part1 = all_digits[0:2]
                part2 = all_digits[2:5]
                part3 = all_digits[5:7]
                letter = letters[0]
                
                return {
                    'full_text': text,
                    'part1': part1,
                    'letter': letter,
                    'part2': part2,
                    'part3': part3,
                    'formatted': f"{part1} {letter} {part2} - {part3}"
                }
            except:
                pass
        
        return None
    
    def process_vehicle(self, image, vehicle_bbox):
        """
        پردازش کامل یک خودرو: تشخیص و خواندن پلاک
        
        Args:
            image: تصویر کامل
            vehicle_bbox: مختصات خودرو (x1, y1, x2, y2)
        
        Returns:
            dict: نتایج پردازش پلاک
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # برش تصویر خودرو
        vehicle_image = image[y1:y2, x1:x2]
        
        if vehicle_image.size == 0:
            return None
        
        # تشخیص ناحیه پلاک
        plates = self.detect_plate_region(vehicle_image)
        
        if not plates:
            return {
                'has_plate': False,
                'plate_count': 0,
                'plates': []
            }
        
        # پردازش هر پلاک
        results = []
        for plate in plates[:2]:  # حداکثر 2 پلاک
            px1, py1, px2, py2 = plate['bbox']
            plate_image = vehicle_image[py1:py2, px1:px2]
            
            # خواندن متن
            text = self.read_plate_text(plate_image)
            
            # تجزیه متن
            parsed = self.parse_iranian_plate(text) if text else None
            
            # محاسبه مختصات در تصویر اصلی
            absolute_bbox = (
                x1 + px1,
                y1 + py1,
                x1 + px2,
                y1 + py2
            )
            
            results.append({
                'bbox': absolute_bbox,
                'relative_bbox': (px1, py1, px2, py2),
                'confidence': plate['confidence'],
                'text': text,
                'parsed': parsed
            })
        
        return {
            'has_plate': len(results) > 0,
            'plate_count': len(results),
            'plates': results
        }


def main():
    """تابع تست"""
    import sys
    
    if len(sys.argv) < 2:
        print("استفاده: python plate_reader.py <image_path>")
        return
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"خطا: نمی‌توان تصویر '{image_path}' را خواند.")
        return
    
    reader = IranianPlateReader()
    
    # فرض می‌کنیم کل تصویر یک خودرو است
    h, w = image.shape[:2]
    result = reader.process_vehicle(image, (0, 0, w, h))
    
    print("\nنتیجه:")
    print(f"پلاک شناسایی شد: {result['has_plate']}")
    print(f"تعداد پلاک: {result['plate_count']}")
    
    for i, plate in enumerate(result['plates'], 1):
        print(f"\nپلاک {i}:")
        print(f"  متن: {plate['text']}")
        if plate['parsed']:
            print(f"  فرمت شده: {plate['parsed']['formatted']}")


if __name__ == "__main__":
    main()
