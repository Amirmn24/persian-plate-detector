"""
ماژول تشخیص و خواندن پلاک ایرانی.
برای تشخیص پلاک از مدل YOLOv5 مخصوص پلاک فارسی (plateYolo.pt) استفاده می‌شود
در صورت نبود فایل مدل، از روش پردازش تصویر (رنگ/شکل) به عنوان fallback استفاده می‌شود.
منبع مدل: https://github.com/truthofmatthew/persian-license-plate-recognition
"""

import os
import cv2
import numpy as np
import re
import time

# مسیر ریشه پروژه (یک سطح بالاتر از utlis)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# مدل‌های پلاک فارسی از repo persian-license-plate-recognition
_PLATE_YOLOV5_PATH = os.path.join(_PROJECT_ROOT, "model", "plateYolo.pt")
_CHARS_YOLO_PATH = os.path.join(_PROJECT_ROOT, "model", "CharsYolo.pt")
_EASYOCR_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "easyocr")

# مجموعه کاراکترهای مجاز پلاک ایران: ۷ رقم + ۱ حرف = ۸ کاراکتر
_PLATE_DIGITS = set("۰۱۲۳۴۵۶۷۸۹")
_PLATE_LETTERS_SET = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
_PLATE_WHITELIST_STR = "۰۱۲۳۴۵۶۷۸۹آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

# نگاشت کلاس مدل CharsYolo به حرف/عدد فارسی (مطابق PLPR)
_CHARS_LABEL_TO_PERSIAN = {
    "0": "۰", "1": "۱", "2": "۲", "3": "۳", "4": "۴",
    "5": "۵", "6": "۶", "7": "۷", "8": "۸", "9": "۹",
    "A": "آ", "B": "ب", "D": "د", "Gh": "ق", "H": "ه", "J": "ج",
    "L": "ل", "M": "م", "N": "ن", "P": "پ", "PuV": "ع", "PwD": "ژ",
    "Sad": "ص", "Sin": "س", "T": "ط", "Taxi": "ت", "V": "و", "Y": "ی",
}
_CHARS_LABEL_MAP = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "D", "Gh", "H", "J", "L", "M", "N", "P",
    "PuV", "PwD", "Sad", "Sin", "T", "Taxi", "V", "Y",
]


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
    
    def __init__(self, plate_model_path=None, confidence=0.25):
        """
        مقداردهی اولیه.
        اگر مدل plateYolo.pt (YOLOv5 پلاک فارسی) در مسیر model/plateYolo.pt باشد، از آن استفاده می‌شود؛
        در غیر این صورت از روش تشخیص بر اساس رنگ/شکل استفاده می‌شود.
        
        Args:
            plate_model_path: مسیر فایل مدل تشخیص پلاک (plateYolo.pt). پیش‌فرض: model/plateYolo.pt
            confidence: حداقل اطمینان برای تشخیص پلاک (۰.۲۵ برای پلاک‌های کوچک مناسب است)
        """
        self.confidence = confidence
        self.ocr_reader = None
        self._ocr_failed = False
        self._plate_model = None
        self._chars_model = None  # مدل YOLOv5 برای تشخیص کاراکتر پلاک (اعداد و حرف فارسی)
        
        path = plate_model_path or _PLATE_YOLOV5_PATH
        if not os.path.isabs(path):
            path = os.path.join(_PROJECT_ROOT, path) if not os.path.isfile(path) else path
        if os.path.isfile(path):
            try:
                import torch
                print(f"[*] در حال بارگذاری مدل تشخیص پلاک YOLOv5 از: {path}")
                self._plate_model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=path,
                    force_reload=False,
                    trust_repo=True,
                )
                self._plate_model.conf = self.confidence
                print("[+] مدل تشخیص پلاک (YOLOv5) آماده است!")
            except Exception as e:
                print(f"[!] خطا در بارگذاری مدل پلاک: {e}. از روش تشخیص با رنگ/شکل استفاده می‌شود.")
                self._plate_model = None
        else:
            print("[!] فایل مدل پلاک (plateYolo.pt) یافت نشد. از روش رنگ/شکل استفاده می‌شود.")
            print("    برای دقت بالاتر، مدل را از این آدرس دریافت کنید:")
            print("    https://github.com/truthofmatthew/persian-license-plate-recognition")
            print("    و در پوشه model با نام plateYolo.pt ذخیره کنید.")
        
        # مدل استخراج کاراکتر (اعداد و حرف فارسی) از ناحیه پلاک
        chars_path = _CHARS_YOLO_PATH
        if not os.path.isabs(chars_path) and not os.path.isfile(chars_path):
            chars_path = os.path.join(_PROJECT_ROOT, chars_path)
        if os.path.isfile(chars_path):
            try:
                import torch
                print("[*] در حال بارگذاری مدل تشخیص کاراکتر پلاک (CharsYolo)...")
                self._chars_model = torch.hub.load(
                    "ultralytics/yolov5", "custom", path=chars_path,
                    force_reload=False, trust_repo=True,
                )
                self._chars_model.conf = 0.2
                print("[+] مدل کاراکتر پلاک آماده است!")
            except Exception as e:
                print(f"[!] مدل CharsYolo بارگذاری نشد: {e}. از OCR برای خواندن متن استفاده می‌شود.")
                self._chars_model = None
    
    @staticmethod
    def _easyocr_models_exist():
        """بررسی وجود فایل‌های مدل EasyOCR در پوشهٔ پروژه تا دانلود مجدد نشود."""
        if not os.path.isdir(_EASYOCR_MODEL_DIR):
            return False
        # EasyOCR حداقل فایل تشخیص (craft) و مدل‌های زبان را ذخیره می‌کند
        for name in os.listdir(_EASYOCR_MODEL_DIR):
            if name.endswith(".pth") or name.endswith(".zip"):
                return True
        return False

    def _get_ocr_reader(self):
        """دریافت یا ایجاد OCR reader (lazy loading). فقط در صورت نبود مدل دانلود می‌شود."""
        if self._ocr_failed:
            return None
        if self.ocr_reader is None:
            try:
                import easyocr
            except ImportError:
                print("[!] خطا: کتابخانه easyocr نصب نیست. pip install easyocr")
                self._ocr_failed = True
                return None
            os.makedirs(_EASYOCR_MODEL_DIR, exist_ok=True)
            # اگر مدل‌ها قبلاً در پوشهٔ پروژه ذخیره شده‌اند، دانلود مجدد نکن
            use_local_only = self._easyocr_models_exist()
            max_attempts = 2 if use_local_only else 3
            for attempt in range(1, max_attempts + 1):
                try:
                    if use_local_only:
                        print("[*] در حال بارگذاری مدل OCR از پوشهٔ محلی (بدون دانلود)...")
                    else:
                        print(f"[*] در حال بارگذاری مدل OCR فارسی (تلاش {attempt}/{max_attempts})...")
                    if attempt > 1:
                        time.sleep(3)
                    self.ocr_reader = easyocr.Reader(
                        ["fa", "en"],
                        gpu=False,
                        model_storage_directory=_EASYOCR_MODEL_DIR,
                        download_enabled=not use_local_only,
                    )
                    print("[+] مدل OCR آماده است!")
                    return self.ocr_reader
                except Exception as e:
                    err_msg = str(e).lower()
                    if use_local_only:
                        # مدل محلی ناقص بود؛ یک بار با دانلود تلاش کن
                        use_local_only = False
                        self.ocr_reader = None
                        continue
                    if "retrieval incomplete" in err_msg or "urlopen" in err_msg or "got only" in err_msg:
                        print(f"[!] دانلود ناقص بود. پاک‌سازی و تلاش مجدد...")
                        self._remove_incomplete_easyocr_models()
                        self.ocr_reader = None
                    else:
                        print(f"[!] خطا در بارگذاری OCR: {e}")
                        self._ocr_failed = True
                        return None
            print("[!] مدل OCR بارگذاری نشد. فقط تشخیص پلاک (بدون خواندن متن) انجام می‌شود.")
            self._ocr_failed = True
        return self.ocr_reader
    
    @staticmethod
    def _remove_incomplete_easyocr_models():
        """پاک کردن فایل‌های ناقص مدل EasyOCR تا در تلاش بعدی دوباره دانلود شوند."""
        if not os.path.isdir(_EASYOCR_MODEL_DIR):
            return
        try:
            for name in os.listdir(_EASYOCR_MODEL_DIR):
                path = os.path.join(_EASYOCR_MODEL_DIR, name)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    for sub in os.listdir(path):
                        os.remove(os.path.join(path, sub))
                    os.rmdir(path)
        except OSError:
            pass
    
    def detect_plate_region(self, vehicle_image, vehicle_bbox=None):
        """
        تشخیص ناحیه پلاک در تصویر برش‌خوردهٔ خودرو.
        اگر مدل YOLOv5 پلاک (plateYolo.pt) بارگذاری شده باشد از آن استفاده می‌شود؛
        در غیر این صورت از روش رنگ/شکل استفاده می‌شود.
        """
        plates = []
        if self._plate_model is not None:
            plates = self._detect_plate_by_yolov5(vehicle_image)
        if not plates:
            plates = self._detect_plate_by_color(vehicle_image)
        return plates
    
    def _detect_plate_by_yolov5(self, image):
        """
        تشخیص پلاک با مدل YOLOv5 آموزش‌دیده برای پلاک فارسی (plateYolo.pt).
        مدل از پروژه persian-license-plate-recognition است.
        """
        if image is None or image.size == 0:
            return []
        try:
            # YOLOv5 از torch.hub تصویر RGB می‌خواهد
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._plate_model(img_rgb)
            # خروجی به صورت pandas با ستون‌های xmin, ymin, xmax, ymax, confidence
            df = results.pandas().xyxy[0]
            plates = []
            for _, row in df.iterrows():
                conf = float(row.get("confidence", 0))
                if conf < self.confidence:
                    continue
                x1 = int(row["xmin"])
                y1 = int(row["ymin"])
                x2 = int(row["xmax"])
                y2 = int(row["ymax"])
                h_img, w_img = image.shape[:2]
                # محدود کردن به محدوده تصویر و کمی حاشیه
                x1 = max(0, x1 - 2)
                y1 = max(0, y1 - 2)
                x2 = min(w_img, x2 + 2)
                y2 = min(h_img, y2 + 2)
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                plates.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "area": area,
                })
            return plates
        except Exception as e:
            print(f"[!] خطا در تشخیص پلاک با YOLOv5: {e}")
            return []
    
    def _detect_plate_by_color(self, image):
        """
        تشخیص پلاک بر اساس رنگ و شکل:
        - تمرکز روی یک‌سوم پایین تصویر (محل معمول پلاک)
        - مستطیل با نسبت ابعاد پلاک ایرانی؛ پلاک می‌تواند کمی کج باشد (minAreaRect)
        - رنگ سفید یا زرد (پلاک سفید/زرد)
        """
        h_img, w_img = image.shape[:2]
        # ناحیه جستجو: نیمهٔ پایین تصویر (پلاک اکثراً پایین است)، ترجیحاً یک‌سوم پایین
        search_y_start = int(h_img * 0.45)  # از ۴۵٪ ارتفاع به پایین
        search_roi = image[search_y_start:, :]
        h_roi, w_roi = search_roi.shape[:2]
        
        # ترکیب چند روش برای ماسک بهتر
        gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        
        # ماسک سفید/خاکستری روشن (پلاک سفید)
        _, white_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = cv2.inRange(gray, 180, 255)
        
        # ماسک HSV برای سفید (اشباع کم، روشنایی بالا)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 55, 255])
        mask_white_hsv = cv2.inRange(hsv, lower_white, upper_white)
        
        # ماسک زرد (پلاک زرد)
        lower_yellow = np.array([15, 60, 120])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask = cv2.bitwise_or(mask_white_hsv, mask_yellow)
        mask = cv2.bitwise_or(mask, bright)
        
        # مورفولوژی: بستن حفره‌ها و حذف نویز
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:  # حداقل اندازه معقول برای پلاک
                continue
            
            # مستطیل با حداقل مساحت تا پلاک کج هم تشخیص داده شود
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw <= 0 or rh <= 0:
                continue
            
            # نسبت ابعاد پلاک ایرانی تقریباً ۳:۱ تا ۴:۱ (عرض به ارتفاع)
            if rw >= rh:
                aspect = rw / rh
            else:
                aspect = rh / rw
            
            if aspect < 2.2 or aspect > 5.5:
                continue
            
            # حداقل اندازه پیکسلی (عرض و ارتفاع واقعی بعد از چرخش)
            box_w = max(rw, rh)
            box_h = min(rw, rh)
            if box_w < 60 or box_h < 18:
                continue
            
            # زاویه مجاز: پلاک تقریباً افقی (حداکثر کجی معمول)
            if abs(angle) > 75 and abs(angle) < 105:
                angle = angle - 90
            if abs(angle) > 25:
                continue
            
            # کادر محاط (برای برش تصویر)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            x_roi, y_roi, w_roi_rect, h_roi_rect = cv2.boundingRect(box_points)
            
            # برگرداندن مختصات به فضای کل تصویر خودرو
            x1 = max(0, x_roi)
            y1 = max(0, search_y_start + y_roi)
            x2 = min(w_img, x_roi + w_roi_rect)
            y2 = min(h_img, search_y_start + y_roi + h_roi_rect)
            
            # حاشیه کم برای اینکه لبه پلاک و حروف بریده نشوند
            pad = max(2, int(min(w_roi_rect, h_roi_rect) * 0.08))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)
            
            # امتیاز: ترجیح ناحیهٔ پایین‌تر (یک‌سوم پایین)
            center_y = (y1 + y2) / 2
            lower_third_bonus = 1.0 if center_y >= h_img * (2/3) else (0.7 if center_y >= h_img * 0.5 else 0.4)
            area_score = min(area / 5000, 1.0)
            aspect_ok = 1.0 if 2.8 <= aspect <= 4.5 else 0.8
            score = lower_third_bonus * 0.5 + area_score * 0.3 + aspect_ok * 0.2
            
            plates.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': min(0.5 + score * 0.4, 0.95),
                'area': area,
                '_score': score,
                'angle': angle,
            })
        
        # مرتب‌سازی بر اساس امتیاز (ترجیح یک‌سوم پایین و اندازه مناسب)
        plates.sort(key=lambda x: (x['_score'], x['area']), reverse=True)
        for p in plates:
            p.pop('_score', None)
        
        return plates
    
    def _read_plate_with_chars_yolo(self, plate_image):
        """
        استخراج اعداد و حرف فارسی از تصویر پلاک با مدل CharsYolo.
        پلاک ایرانی معمولاً راست‌به‌چپ است؛ هر دو ترتیب (چپ→راست و راست→چپ) امتحان می‌شود.
        آستانه اطمینان برای حروف کمتر است تا حرف پلاک حذف نشود.
        """
        if self._chars_model is None or plate_image is None or plate_image.size == 0:
            return None
        try:
            h, w = plate_image.shape[:2]
            target_h = 132
            target_w = max(320, int(w * target_h / max(h, 1)))
            img = cv2.resize(plate_image, (target_w, target_h))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self._chars_model(img_rgb)
            pred = results.pred[0]
            if pred is None or len(pred) == 0:
                return None
            # ارقام: آستانه بالاتر؛ حروف (کلاس ۱۰ به بعد): آستانه پایین‌تر تا ع/پ/ت و غیره حذف نشوند
            PERSIAN_DIGITS = set("۰۱۲۳۴۵۶۷۸۹")
            dets = []
            for i in range(pred.shape[0]):
                x1, y1, x2, y2 = pred[i, :4].tolist()
                conf = float(pred[i, 4])
                cls = int(pred[i, 5])
                if cls < len(_CHARS_LABEL_MAP):
                    label = _CHARS_LABEL_MAP[cls]
                    persian_char = _CHARS_LABEL_TO_PERSIAN.get(label, label)
                    is_digit = persian_char in PERSIAN_DIGITS
                    if is_digit and conf < 0.35:
                        continue
                    if not is_digit and conf < 0.08:
                        continue
                    cx = (x1 + x2) / 2
                    dets.append((cx, persian_char))
            if not dets:
                return None
            dets.sort(key=lambda x: x[0])
            text_ltr = "".join(c for _, c in dets)
            text_rtl = "".join(c for _, c in reversed(dets))
            has_letter_ltr = any(c not in PERSIAN_DIGITS for c in text_ltr)
            has_letter_rtl = any(c not in PERSIAN_DIGITS for c in text_rtl)
            if has_letter_rtl:
                text = text_rtl
            elif has_letter_ltr:
                text = text_ltr
            else:
                text = text_rtl if len(text_rtl) >= len(text_ltr) else text_ltr
            # اگر فقط رقم آمد، یک بار با آستانهٔ خیلی پایین فقط برای حرف امتحان کن
            if text and not any(c not in PERSIAN_DIGITS for c in text) and len(dets) >= 5:
                old_conf = self._chars_model.conf
                try:
                    self._chars_model.conf = 0.08
                    res2 = self._chars_model(img_rgb)
                    pred2 = res2.pred[0]
                    if pred2 is not None and pred2.shape[0] > 0:
                        x_min_all = min(d[0] for d in dets)
                        x_max_all = max(d[0] for d in dets)
                        for i in range(pred2.shape[0]):
                            x1, y1, x2, y2 = pred2[i, :4].tolist()
                            conf = float(pred2[i, 4])
                            cls = int(pred2[i, 5])
                            if conf < 0.08 or cls >= len(_CHARS_LABEL_MAP):
                                continue
                            label = _CHARS_LABEL_MAP[cls]
                            persian_char = _CHARS_LABEL_TO_PERSIAN.get(label, label)
                            if persian_char in PERSIAN_DIGITS:
                                continue
                            cx = (x1 + x2) / 2
                            if x_min_all <= cx <= x_max_all:
                                dets.append((cx, persian_char))
                                dets.sort(key=lambda x: x[0])
                                text_rtl = "".join(c for _, c in reversed(dets))
                                text_ltr = "".join(c for _, c in dets)
                                text = text_rtl if len(text_rtl) >= len(text_ltr) else text_ltr
                                break
                finally:
                    self._chars_model.conf = old_conf
            return text if len(text) >= 5 else None
        except Exception as e:
            print(f"[!] خطا در استخراج کاراکتر پلاک: {e}")
            return None

    def _normalize_to_persian_digits(self, text):
        """تبدیل ارقام انگلیسی به فارسی در متن."""
        if not text:
            return text
        en_to_fa = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
        return text.translate(en_to_fa)

    @staticmethod
    def _crop_letter_region(plate_image, margin_left=0.30, margin_right=0.50):
        """
        برش ناحیهٔ افقی محل حرف وسط پلاک ایرانی (۲ رقم + حرف + ۳ رقم + ۲ رقم).
        حرف تقریباً در یک‌چهارم تا نیم عرض پلاک قرار دارد (بسته به راست/چپ).
        """
        if plate_image is None or plate_image.size == 0:
            return None
        h, w = plate_image.shape[:2]
        if w < 15:
            return None
        x1 = int(w * margin_left)
        x2 = int(w * margin_right)
        x1 = max(0, min(x1, w - 5))
        x2 = max(x1 + 5, min(x2, w))
        return plate_image[:, x1:x2]

    @staticmethod
    def _get_letter_region_crops(plate_image):
        """
        چند برش افقی از ناحیهٔ احتمالی حرف برای امتحان با OCR.
        با چند پنجرهٔ کمی جابه‌جا شده احتمال تشخیص حرف را بالا می‌بریم.
        """
        crops = []
        # پنجره‌های مختلف برای محل حرف (پلاک راست‌به‌چپ: حرف بعد از دو رقم اول)
        for (left, right) in [(0.28, 0.48), (0.32, 0.52), (0.25, 0.55), (0.30, 0.50)]:
            c = IranianPlateReader._crop_letter_region(plate_image, left, right)
            if c is not None and c.size > 0:
                crops.append(c)
        return crops

    @staticmethod
    def _preprocess_for_persian_letter(image):
        """
        پیش‌پردازش تصویر برای بهبود تشخیص حرف فارسی (مثل Persian-OCR).
        شامل: گرِیسکیل، بزرگ‌نمایی، حذف نویز، آستانه‌گذاری.
        """
        if image is None or image.size == 0:
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape[:2]
        # بزرگ‌نمایی برای خوانایی بهتر حرف (مثل Persian-OCR با ضریب 1024)
        scale = max(2.0, 384.0 / max(w, 1))
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # حذف نویز (median blur مثل Persian-OCR)
        gray = cv2.medianBlur(gray, 3)
        # آستانه‌گذاری برای متن سیاه روی پس‌زمینه سفید (پلاک)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def _preprocess_letter_for_ocr(image, target_height=80):
        """
        پیش‌پردازش قوی برای یک ناحیهٔ تک‌حرف: ارتفاع ثابت، کنتراست بالا، مناسب OCR.
        """
        if image is None or image.size == 0:
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape[:2]
        scale = target_height / max(h, 1)
        new_w = max(20, int(w * scale))
        gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh

    @staticmethod
    def _preprocess_plate_farsi_ocr(image):
        """
        پیش‌پردازش تصویر پلاک به سبک FarsiOCR برای بهبود OCR فارسی.
        منبع: https://github.com/stafazzoli/FarsiOCR (resize، گرِیسکیل، مورفولوژی، آستانه).
        """
        if image is None or image.size == 0:
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape[:2]
        # بزرگ‌نمایی برای Tesseract (مثل FarsiOCR با عرض ۱۸۰۰؛ برای پلاک کوچک ۸۰۰ کافی است)
        target_width = max(800, min(1200, w * 3))
        scale = target_width / max(w, 1)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # حذف نویز و صاف‌سازی (مثل FarsiOCR: open + close + bitwise_or)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        gray = cv2.bitwise_or(gray, closing)
        # آستانه OTSU برای متن سیاه روی پس‌زمینه سفید (بهبود خوانایی حرف)
        _, thresh = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh

    def _read_full_plate_8_chars_with_tesseract(self, plate_image):
        """
        خواندن کل رشته ۸ کاراکتری پلاک با Tesseract (فارسی) و whitelist رقم+حرف.
        خروجی: (رشته_۸_کاراکتری, حرف) یا (None, None).
        مثل FarsiOCR: پیش‌پردازش + pytesseract با lang=fas.
        """
        try:
            import pytesseract
        except ImportError:
            return None, None
        # whitelist: فقط ارقام و حروف مجاز پلاک تا نویز حذف شود
        config = f'-l fas --psm 7 -c tessedit_char_whitelist="{_PLATE_WHITELIST_STR}"'
        preprocessed = self._preprocess_plate_farsi_ocr(plate_image)
        if preprocessed is None:
            return None, None
        try:
            text = pytesseract.image_to_string(preprocessed, config=config)
        except Exception:
            return None, None
        return self._parse_8_char_plate_string(text or "")

    @staticmethod
    def _parse_8_char_plate_string(raw):
        """
        از خروجی OCR فقط کاراکترهای مجاز پلاک (رقم/حرف فارسی) را به ترتیب برمی‌دارد.
        اگر دقیقاً ۷ رقم و ۱ حرف در یک رشته ۸ کاراکتری باشد، برمی‌گرداند (رشته_۸_کاراکتری, حرف).
        ارقام انگلیسی به فارسی تبدیل می‌شوند.
        """
        if not raw:
            return None, None
        en_to_fa = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
        raw = raw.translate(en_to_fa)
        allowed = [c for c in raw if c in _PLATE_DIGITS or c in _PLATE_LETTERS_SET]
        if len(allowed) < 8:
            digits = [c for c in allowed if c in _PLATE_DIGITS]
            letters = [c for c in allowed if c in _PLATE_LETTERS_SET]
            if len(digits) == 7 and len(letters) == 1:
                return "".join(allowed), letters[0]
            return None, None
        # پنجره ۸ کاراکتری با ۷ رقم و ۱ حرف
        for i in range(len(allowed) - 7):
            window = allowed[i : i + 8]
            digits = [c for c in window if c in _PLATE_DIGITS]
            letters = [c for c in window if c in _PLATE_LETTERS_SET]
            if len(digits) == 7 and len(letters) == 1:
                return "".join(window), letters[0]
        return None, None

    def _read_full_plate_8_chars_with_easyocr(self, plate_image):
        """
        خواندن کل رشته ۸ کاراکتری با EasyOCR روی تصویر پیش‌پردازش‌شده (FarsiOCR style).
        خروجی: (رشته_۸_کاراکتری, حرف) یا (None, None).
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return None, None
        preprocessed = self._preprocess_plate_farsi_ocr(plate_image)
        if preprocessed is None:
            return None, None
        try:
            results = reader.readtext(preprocessed)
        except Exception:
            return None, None
        # همه متن‌های تشخیص‌داده‌شده را به ترتیب موقعیت (چپ به راست) وصل کن
        if not results:
            return None, None
        # مرتب‌سازی بر اساس موقعیت افقی (مرکز باکس)
        def center_x(item):
            (box, _, _) = item
            return (box[0][0] + box[2][0]) / 2
        results = sorted(results, key=center_x)
        raw = "".join(t for (_, t, _) in results)
        return self._parse_8_char_plate_string(self._normalize_to_persian_digits(raw))

    def _extract_letter_with_persian_ocr_repo(self, plate_image):
        """
        استخراج حرف با مدل CNN از پروژه Persian-letter-OCR در صورت وجود.
        https://github.com/XSharifi/Persian-letter-OCR
        ورودی مدل: تصویر ۲۸×۲۸ گرِیسکیل (حرف دست‌نویس/چاپی).
        در صورت نصب ماژول persian_letter_ocr_helper و وجود وزن‌ها فراخوانی می‌شود.
        """
        try:
            from .persian_letter_ocr_helper import predict_letter_from_plate_region
        except ImportError:
            return None
        letter_region = self._crop_letter_region(plate_image, 0.30, 0.50)
        if letter_region is None or letter_region.size == 0:
            return None
        return predict_letter_from_plate_region(letter_region)

    def _extract_letter_from_ocr(self, plate_image):
        """
        وقتی CharsYolo حرف نیاورده، حرف پلاک را با OCR استخراج می‌کنیم.
        فرمت پلاک ایران: ۲ رقم + حرف + ۳ رقم + ۲ رقم. حرف در وسط پلاک است.
        اول چند برش دقیق از ناحیهٔ حرف امتحان می‌شود (بهبود تشخیص حرف وسط).
        """
        PLATE_LETTERS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
        # ۱) برش‌های ناحیهٔ حرف با پیش‌پردازش قوی تک‌حرف + EasyOCR فقط حروف
        reader = self._get_ocr_reader()
        if reader is not None:
            try:
                for crop in self._get_letter_region_crops(plate_image):
                    processed = self._preprocess_letter_for_ocr(crop, target_height=80)
                    if processed is None:
                        continue
                    for (_, ocr_text, conf) in reader.readtext(processed):
                        if conf < 0.15:
                            continue
                        for c in ocr_text:
                            if c in PLATE_LETTERS:
                                return c
            except Exception:
                pass
        # ۲) همان برش‌ها با پیش‌پردازش سبک Persian-OCR
        if reader is not None:
            try:
                for crop in self._get_letter_region_crops(plate_image):
                    processed = self._preprocess_for_persian_letter(crop)
                    if processed is None:
                        continue
                    for (_, ocr_text, conf) in reader.readtext(processed):
                        if conf < 0.2:
                            continue
                        for c in ocr_text:
                            if c in PLATE_LETTERS:
                                return c
            except Exception:
                pass
        # ۳) پیش‌پردازش کل پلاک و OCR (حرف از کل متن)
        if reader is not None:
            try:
                processed = self._preprocess_for_persian_letter(plate_image)
                if processed is not None:
                    for (_, ocr_text, conf) in reader.readtext(processed):
                        if conf < 0.2:
                            continue
                        for c in ocr_text:
                            if c in PLATE_LETTERS:
                                return c
            except Exception:
                pass
        # ۴) برش ناحیه وسط (روش قبلی)
        if reader is not None:
            try:
                h, w = plate_image.shape[:2]
                if w >= 20:
                    center_roi = plate_image[:, int(w * 0.25) : int(w * 0.75)]
                    processed_center = self._preprocess_for_persian_letter(center_roi)
                    if processed_center is not None:
                        for (_, ocr_text, conf) in reader.readtext(processed_center):
                            if conf < 0.2:
                                continue
                            for c in ocr_text:
                                if c in PLATE_LETTERS:
                                    return c
            except Exception:
                pass
        # ۵) پیش‌پردازش عمومی پلاک و OCR
        if reader is not None:
            try:
                processed = self._preprocess_plate_image(plate_image)
                for (_, ocr_text, conf) in reader.readtext(processed):
                    if conf < 0.2:
                        continue
                    for c in ocr_text:
                        if c in PLATE_LETTERS:
                            return c
            except Exception:
                pass
        # ۶) Tesseract فقط حروف (برش ناحیه حرف + کل)
        letter_tesseract = self._extract_letter_with_tesseract(plate_image)
        if letter_tesseract:
            return letter_tesseract
        # ۷) اختیاری: مدل CNN از پروژه Persian-letter-OCR
        letter_cnn = self._extract_letter_with_persian_ocr_repo(plate_image)
        if letter_cnn:
            return letter_cnn
        return None

    def _extract_letter_with_tesseract(self, plate_image):
        """
        استخراج حرف با Tesseract و whitelist حروف فارسی (مثل Persian-OCR).
        اول روی برش‌های ناحیهٔ حرف، بعد ناحیه وسط و کل پلاک.
        """
        try:
            import pytesseract
        except ImportError:
            return None
        config_letters = r'-l fas --psm 7 -c tessedit_char_whitelist="آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"'
        PLATE_LETTERS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
        try:
            # اول برش‌های دقیق ناحیهٔ حرف با پیش‌پردازش قوی تک‌حرف
            for crop in self._get_letter_region_crops(plate_image):
                img = self._preprocess_letter_for_ocr(crop, target_height=80)
                if img is None:
                    continue
                text = pytesseract.image_to_string(img, config=config_letters)
                for c in (text or "").strip():
                    if c in PLATE_LETTERS:
                        return c
            # بعد پیش‌پردازش سبک روی همان برش‌ها
            for crop in self._get_letter_region_crops(plate_image):
                img = self._preprocess_for_persian_letter(crop)
                if img is None:
                    continue
                text = pytesseract.image_to_string(img, config=config_letters)
                for c in (text or "").strip():
                    if c in PLATE_LETTERS:
                        return c
            # کل پلاک
            img = self._preprocess_for_persian_letter(plate_image)
            if img is not None:
                text = pytesseract.image_to_string(img, config=config_letters)
                for c in (text or "").strip():
                    if c in PLATE_LETTERS:
                        return c
            # ناحیه وسط
            h, w = plate_image.shape[:2]
            if w >= 20:
                center_roi = plate_image[:, int(w * 0.25) : int(w * 0.75)]
                img_center = self._preprocess_for_persian_letter(center_roi)
                if img_center is not None:
                    text = pytesseract.image_to_string(img_center, config=config_letters)
                    for c in (text or "").strip():
                        if c in PLATE_LETTERS:
                            return c
        except Exception:
            pass
        return None

    def read_plate_text(self, plate_image):
        """
        استخراج متن پلاک (اعداد و حرف فارسی).
        در صورت وجود مدل CharsYolo از آن استفاده می‌شود، وگرنه از EasyOCR.
        اگر CharsYolo فقط ۷ رقم برگرداند، با EasyOCR سعی می‌کنیم حرف را استخراج کنیم.
        """
        text = None
        if self._chars_model is not None:
            text = self._read_plate_with_chars_yolo(plate_image)
        if not text:
            reader = self._get_ocr_reader()
            if reader is not None:
                processed = self._preprocess_plate_image(plate_image)
                try:
                    results = reader.readtext(processed)
                    if results:
                        texts = [t for (_, t, c) in results if c > 0.3]
                        text = " ".join(texts) if texts else None
                except Exception as e:
                    print(f"[!] خطا در OCR: {e}")
        if text:
            text = self._normalize_to_persian_digits(text.strip())
        # اگر فقط ۷ رقم داریم و حرفی نیست: اول کل رشته ۸ کاراکتری را با OCR بخوان (مثل FarsiOCR)، بعد حرف تنها
        if text:
            digits = re.findall(r"[۰-۹0-9]", text)
            letters = re.findall(r"[آ-ی]", text)
            if len(digits) == 7 and len(letters) == 0:
                # ۱) خواندن کل پلاک ۸ کاراکتری با Tesseract (پیش‌پردازش FarsiOCR)
                full_str, letter = self._read_full_plate_8_chars_with_tesseract(plate_image)
                if full_str:
                    text = full_str
                else:
                    # ۲) همان کار با EasyOCR روی تصویر پیش‌پردازش‌شده
                    full_str, letter = self._read_full_plate_8_chars_with_easyocr(plate_image)
                    if full_str:
                        text = full_str
                    else:
                        # ۳) فقط استخراج حرف (روش قبلی)
                        letter = self._extract_letter_from_ocr(plate_image)
                        if letter:
                            rev = digits[::-1]
                            text = "".join(rev[5:7]) + letter + "".join(rev[2:5]) + "".join(rev[0:2])
        return text
    
    def _preprocess_plate_image(self, image):
        """
        پیش‌پردازش تصویر پلاک برای بهبود OCR (شامل اصلاح کجی احتمالی).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # اصلاح کجی: تخمین زاویه از خطوط افقی متن پلاک
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=min(w, h) // 4, minLineLength=w // 4, maxLineGap=10)
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > 5:
                    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(ang) < 15:
                        angles.append(ang)
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # بزرگ‌نمایی برای OCR بهتر
        scale_factor = 2
        gray = cv2.resize(gray, (w * scale_factor, h * scale_factor))
        
        # threshold و کاهش نویز
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def parse_iranian_plate(self, text):
        """
        تجزیه متن پلاک ایرانی و استخراج: ۲ رقم + حرف فارسی + ۳ رقم + ۲ رقم.
        ارقام و حرف خروجی همیشه به صورت فارسی (۰-۹ و آ-ی) برمی‌گردند.
        """
        if not text:
            return None
        text = text.strip()
        # نرمال کردن به فارسی برای الگو
        text_norm = self._normalize_to_persian_digits(text)
        # حذف فاصله و نویز
        text_clean = re.sub(r"\s+", "", text_norm)
        
        # الگو: ۲ رقم + یک حرف فارسی + ۳ رقم + ۲ رقم (ارقام می‌توانند فارسی یا انگلیسی باشند)
        # [۰-۹0-9] برای رقم، [آ-ی] برای حرف
        pattern = r"([۰-۹0-9]{2})\s*([آ-ی])\s*([۰-۹0-9]{3})\s*([۰-۹0-9]{2})"
        match = re.search(pattern, text_norm)
        if not match:
            match = re.search(
                r"([۰-۹0-9]{2})([آ-ی])([۰-۹0-9]{3})([۰-۹0-9]{2})",
                text_clean,
            )
        if match:
            part1 = self._normalize_to_persian_digits(match.group(1))
            letter = match.group(2)
            part2 = self._normalize_to_persian_digits(match.group(3))
            part3 = self._normalize_to_persian_digits(match.group(4))
            formatted = f"{part1} {letter} {part2} - {part3}"
            return {
                "full_text": text_norm,
                "part1": part1,
                "letter": letter,
                "part2": part2,
                "part3": part3,
                "formatted": formatted,
            }
        # الگوی خوانش چپ→راست: ۲ رقم + ۳ رقم + حرف + ۲ رقم (مثلاً ۳۷۷۸۳ع۹۱) → خروجی استاندارد ۹۱ ع ۷۸۳ - ۳۷
        if len(text_clean) == 8:
            digits = re.findall(r"[۰-۹0-9]", text_clean)
            letters = re.findall(r"[آ-ی]", text_clean)
            if len(digits) == 7 and len(letters) == 1:
                pos = text_clean.index(letters[0])
                if pos == 5:
                    part3 = self._normalize_to_persian_digits(text_clean[0:2])
                    part2 = self._normalize_to_persian_digits(text_clean[2:5])
                    letter = letters[0]
                    part1 = self._normalize_to_persian_digits(text_clean[6:8])
                    formatted = f"{part1} {letter} {part2} - {part3}"
                    return {
                        "full_text": text_norm,
                        "part1": part1,
                        "letter": letter,
                        "part2": part2,
                        "part3": part3,
                        "formatted": formatted,
                    }
        return self._parse_flexible(text_norm)
    
    def _parse_flexible(self, text):
        """
        تجزیه ۷ رقم (+ در صورت وجود حرف) و نمایش همیشه از چپ به راست: ۱۲ ط ۲۵۲ - ۵۱.
        اگر مدل ترتیب را برعکس داده باشد، کل رشتهٔ ۷رقمی را یک‌بار وارون می‌کنیم،
        بعد به صورت ۲ + ۳ + ۲ تقسیم می‌کنیم تا هر سه بخش (از جمله بخش ۳رقمی) درست نمایش داده شوند.
        """
        digits = re.findall(r"[۰-۹0-9]", text)
        letters = re.findall(r"[آ-ی]", text)
        all_digits = "".join(digits)
        all_digits = self._normalize_to_persian_digits(all_digits)
        if len(all_digits) < 7:
            return None
        letter = letters[0] if letters else "?"
        # وارون کردن کل ۷ رقم تا نمایش از چپ به راست درست شود (۱۲، ۲۵۲، ۵۱)
        rev = all_digits[::-1]
        part1 = rev[5:7]
        part2 = rev[2:5]
        part3 = rev[0:2]
        formatted = f"{part1} {letter} {part2} - {part3}"
        return {
            "full_text": text,
            "part1": part1,
            "letter": letter,
            "part2": part2,
            "part3": part3,
            "formatted": formatted,
        }
    
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
