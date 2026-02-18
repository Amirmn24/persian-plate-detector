"""
ماژول اختیاری برای تشخیص حرف فارسی با مدل CNN پروژه Persian-letter-OCR.
https://github.com/XSharifi/Persian-letter-OCR

استفاده:
  - پروژه را کلون کنید و طبق README مدل را آموزش دهید (یا وزن‌های از پیش آموزش‌دیده را در result/ قرار دهید).
  - مسیر پوشه result/ (شامل W1.txt, W2.txt, B1.txt, B2.txt, convW.txt, convBias.txt) را در متغیر محیطی
    PERSIAN_LETTER_OCR_RESULT قرار دهید، یا پوشه result را در همین پروژه در model/persian_letter_ocr/ بگذارید.
  - در صورت نبود وزن‌ها یا TensorFlow، این ماژول None برمی‌گرداند و بقیهٔ pipeline (OCR) استفاده می‌شود.
"""

import os
import cv2
import numpy as np

# مسیر ریشه پروژه
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# مسیر پیش‌فرض وزن‌های Persian-letter-OCR
_DEFAULT_WEIGHT_DIR = os.path.join(_PROJECT_ROOT, "model", "persian_letter_ocr", "result")

# نگاشت کلاس‌های مدل (Hoda dataset - ReadDataset.maplabel) به حرف فارسی یونیکد
# منبع: https://github.com/XSharifi/Persian-letter-OCR (ReadDataset.py)
_PERSIAN_LETTER_OCR_CLASS_TO_CHAR = {
    0: "آ",   # alef
    1: "ب",   # be
    2: "پ",   # pe
    3: "ت",   # te
    4: "ث",   # the
    5: "ج",   # jim
    6: "چ",   # che
    7: "ح",   # he
    8: "خ",   # khe
    9: "د",   # dal
    10: "ذ",  # zal
    11: "ر",  # re
    12: "ز",  # ze
    13: "ژ",  # zhe
    14: "س",  # sin
    15: "ش",  # shin
    16: "ص",  # sad
    17: "ض",  # zad
    18: "ط",  # ta
    19: "ظ",  # za
    20: "ع",  # ain
    21: "غ",  # ghain
    22: "ف",  # fe
    23: "ق",  # ghe
    24: "ک",  # kaf
    25: "گ",  # gaf
    26: "ل",  # lam
    27: "م",  # mim
    28: "ن",  # noon
    29: "و",  # vav
    30: "ه",  # ha
    31: "ی",  # ya
    32: "آ",  # hamze -> alef
    33: "آ",  # alef-hat
    34: "ه",  # ha-bein
    35: "ه",  # ha-end
}

# حروف مجاز روی پلاک ایران (برای فیلتر خروجی)
_PLATE_LETTERS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")


def _find_result_dir():
    """پیدا کردن پوشهٔ result حاوی وزن‌های مدل."""
    env_path = os.environ.get("PERSIAN_LETTER_OCR_RESULT", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path
    if os.path.isdir(_DEFAULT_WEIGHT_DIR):
        return _DEFAULT_WEIGHT_DIR
    # اگر پروژه Persian-letter-OCR در کنار plate-detector کلون شده
    sibling = os.path.join(os.path.dirname(_PROJECT_ROOT), "Persian-letter-OCR", "result")
    if os.path.isdir(sibling):
        return sibling
    return None


def _required_weight_files(result_dir):
    """بررسی وجود فایل‌های وزن لازم."""
    names = ("W1.txt", "W2.txt", "B1.txt", "B2.txt", "convW.txt", "convBias.txt")
    for n in names:
        if not os.path.isfile(os.path.join(result_dir, n)):
            return False
    return True


def _preprocess_to_28x28(image):
    """
    تبدیل ناحیهٔ حرف به فرمت ورودی مدل Persian-letter-OCR: 28x28 گرِیسکیل، مقدار 0/1 نرمال.
    مشابه ReadDataset: تصویر سیاه روی سفید، resize به 28x28.
    """
    if image is None or image.size == 0:
        return None
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # نرمال 0-1 مثل دیتاست (پس‌زمینه 0، حرف 1)
    flat = (thresh.astype(np.float32) / 255.0).flatten()
    return flat.reshape(1, -1)


# سنسور CNN و session به صورت lazy برای جلوگیری از import سنگین
_cnn_session = None
_cnn_predict_fn = None


def _build_cnn_and_load_weights(result_dir):
    """
    ساخت گراف CNN مطابق CNN.py پروژه Persian-letter-OCR و بارگذاری وزن‌ها.
    از TensorFlow 1.x یا compat.v1 استفاده می‌کند.
    """
    global _cnn_session, _cnn_predict_fn
    if _cnn_predict_fn is not None:
        return True
    try:
        tf = __import__("tensorflow", fromlist=["placeholder", "Variable", "Session"])
        if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
            tf = tf.compat.v1
    except ImportError:
        return False
    if not _required_weight_files(result_dir):
        return False
    try:
        path = result_dir
        W1 = np.loadtxt(os.path.join(path, "W1.txt"), dtype=np.float32)
        W2 = np.loadtxt(os.path.join(path, "W2.txt"), dtype=np.float32)
        B1 = np.loadtxt(os.path.join(path, "B1.txt"), dtype=np.float32)
        B2 = np.loadtxt(os.path.join(path, "B2.txt"), dtype=np.float32)
        convW = np.loadtxt(os.path.join(path, "convW.txt"), dtype=np.float32)
        convB = np.loadtxt(os.path.join(path, "convBias.txt"), dtype=np.float32)
        # در CNN.py: valweight = reshape(convWeights, (-1, layer_size*5*5)) => (4, 25)
        if convW.size == 100:
            convW = convW.reshape(4, 25).reshape(4, 5, 5).transpose(1, 2, 0)  # (5,5,4)
            convW = np.expand_dims(convW, axis=2).astype(np.float32)  # (5,5,1,4)
        else:
            convW = np.reshape(convW, (5, 5, 1, 4)).astype(np.float32)
    except Exception:
        return False
    try:
        X = tf.placeholder(tf.float32, [None, 28 * 28], name="X")
        x_shaped = tf.reshape(X, [-1, 28, 28, 1])
        # Conv: 5x5, 4 filters, SAME, then ReLU, then maxpool 2x2
        conv_out = tf.nn.conv2d(x_shaped, tf.constant(convW), [1, 1, 1, 1], padding="SAME")
        conv_out = tf.nn.bias_add(conv_out, tf.constant(convB))
        conv_out = tf.nn.relu(conv_out)
        pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        flattened = tf.reshape(pool_out, [-1, 14 * 14 * 4])
        hidden = tf.nn.relu(tf.matmul(flattened, tf.constant(W1)) + tf.constant(B1))
        logits = tf.matmul(hidden, tf.constant(W2)) + tf.constant(B2)
        y_pred = tf.argmax(logits, axis=1)
        _cnn_session = tf.Session()
        _cnn_predict_fn = lambda x: _cnn_session.run(y_pred, feed_dict={X: x})
        return True
    except Exception:
        return False


def predict_letter_from_plate_region(letter_region_image):
    """
    پیش‌بینی یک حرف فارسی از تصویر ناحیهٔ حرف پلاک با مدل CNN پروژه Persian-letter-OCR.

    Args:
        letter_region_image: تصویر برش‌خوردهٔ ناحیهٔ حرف (BGR یا گرِیسکیل).

    Returns:
        یک کاراکتر فارسی از حروف مجاز پلاک، یا None در صورت خطا/نبود مدل.
    """
    result_dir = _find_result_dir()
    if result_dir is None or not _required_weight_files(result_dir):
        return None
    if not _build_cnn_and_load_weights(result_dir):
        return None
    x = _preprocess_to_28x28(letter_region_image)
    if x is None:
        return None
    try:
        pred = _cnn_predict_fn(x)
        idx = int(pred[0])
        char = _PERSIAN_LETTER_OCR_CLASS_TO_CHAR.get(idx)
        if char and char in _PLATE_LETTERS:
            return char
        return char  # حتی اگر خارج از پلاک باشد برگردان (فیلتر در plate_reader)
    except Exception:
        return None
