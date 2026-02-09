@echo off
echo ========================================
echo سیستم تشخیص خودرو و خواندن پلاک ایرانی
echo Iranian Vehicle and Plate Detection System
echo ========================================
echo.

echo [1/5] در حال بررسی Python...
python --version
if errorlevel 1 (
    echo خطا: Python نصب نشده است!
    echo لطفا Python 3.8 یا بالاتر را نصب کنید.
    pause
    exit /b 1
)
echo.

echo [2/5] در حال نصب کتابخانه‌ها...
pip install -r requirements.txt
if errorlevel 1 (
    echo خطا در نصب کتابخانه‌ها!
    pause
    exit /b 1
)
echo.

echo [3/5] در حال ساخت دیتابیس...
python manage.py makemigrations
python manage.py migrate
if errorlevel 1 (
    echo خطا در ساخت دیتابیس!
    pause
    exit /b 1
)
echo.

echo [4/5] در حال دانلود مدل YOLOv8...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
echo.

echo [5/5] نصب با موفقیت انجام شد!
echo.
echo برای اجرای برنامه از دستور زیر استفاده کنید:
echo     python manage.py runserver 127.0.0.1:8000
echo.
echo سپس مرورگر خود را باز کرده و به آدرس زیر بروید:
echo     http://127.0.0.1:8000
echo.
pause
