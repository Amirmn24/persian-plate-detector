@echo off
echo ========================================
echo سیستم تشخیص خودرو و خواندن پلاک ایرانی
echo Iranian Vehicle and Plate Detection System
echo ========================================
echo.
echo در حال اجرای سرور توسعه...
echo Server is starting...
echo.
echo مرورگر خود را باز کرده و به آدرس زیر بروید:
echo Open your browser and go to:
echo     http://127.0.0.1:8000
echo.
echo برای توقف سرور از Ctrl+C استفاده کنید
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python manage.py runserver 127.0.0.1:8000
