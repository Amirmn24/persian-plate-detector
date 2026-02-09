import os
import cv2
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .models import UploadedImage
from .forms import ImageUploadForm


# ایجاد یک نمونه از detector و plate_reader به صورت global برای استفاده مجدد
detector = None
plate_reader = None


def get_detector():
    """دریافت یا ایجاد detector"""
    global detector
    if detector is None:
        # Import فقط وقتی که واقعا نیاز باشه
        from utlis.vehicle_detector import VehicleDetector
        detector = VehicleDetector(model_name="yolov8n.pt", confidence=0.5)
    return detector


def get_plate_reader():
    """دریافت یا ایجاد plate reader"""
    global plate_reader
    if plate_reader is None:
        from utlis.plate_reader import IranianPlateReader
        plate_reader = IranianPlateReader(confidence=0.3)
    return plate_reader


def home(request):
    """صفحه اصلی - آپلود و تشخیص تصویر"""
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # ذخیره تصویر
            uploaded_image = form.save()
            
            # مسیر فایل آپلود شده
            image_path = uploaded_image.image.path
            
            # تشخیص خودرو
            try:
                det = get_detector()
                result = det.detect_vehicles(image_path)
                
                if result:
                    # ذخیره نتایج در دیتابیس
                    uploaded_image.has_vehicle = result['has_vehicle']
                    uploaded_image.vehicle_count = result['vehicle_count']
                    uploaded_image.detection_result = result
                    
                    # رسم کادر دور خودروها و تشخیص پلاک
                    if result['has_vehicle']:
                        # بارگذاری تصویر برای پردازش
                        image = cv2.imread(image_path)
                        
                        # تشخیص پلاک برای هر خودرو
                        reader = get_plate_reader()
                        all_plates = []
                        
                        for vehicle in result['vehicles']:
                            x1, y1, x2, y2 = vehicle['bbox']
                            
                            # پردازش پلاک
                            plate_result = reader.process_vehicle(image, (x1, y1, x2, y2))
                            
                            if plate_result and plate_result['has_plate']:
                                vehicle['plate_info'] = plate_result
                                all_plates.extend(plate_result['plates'])
                        
                        # ذخیره اطلاعات پلاک
                        uploaded_image.has_plate = len(all_plates) > 0
                        uploaded_image.plate_count = len(all_plates)
                        uploaded_image.plate_data = all_plates
                        uploaded_image.detection_result = result
                        
                        # ساخت مسیر خروجی
                        processed_dir = os.path.join(
                            settings.MEDIA_ROOT, 
                            'processed',
                            uploaded_image.uploaded_at.strftime('%Y/%m/%d')
                        )
                        os.makedirs(processed_dir, exist_ok=True)
                        
                        filename = os.path.basename(image_path)
                        processed_path = os.path.join(processed_dir, f'detected_{filename}')
                        
                        # رسم کادرها
                        colors = {
                            "car": (0, 255, 0),
                            "motorcycle": (255, 0, 0),
                            "bus": (0, 165, 255),
                            "truck": (0, 0, 255),
                        }
                        
                        for vehicle in result['vehicles']:
                            x1, y1, x2, y2 = vehicle['bbox']
                            color = colors.get(vehicle['type'], (255, 255, 255))
                            label = f"{vehicle['type']} ({vehicle['confidence']:.0%})"
                            
                            # رسم کادر خودرو
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                            
                            # رسم برچسب
                            label_size, _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                            )
                            cv2.rectangle(
                                image,
                                (x1, y1 - label_size[1] - 15),
                                (x1 + label_size[0] + 10, y1),
                                color,
                                -1,
                            )
                            cv2.putText(
                                image, label, (x1 + 5, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            )
                        
                        # رسم کادر پلاک‌ها
                        for plate in all_plates:
                            px1, py1, px2, py2 = plate['bbox']
                            # رسم کادر پلاک (قرمز)
                            cv2.rectangle(image, (px1, py1), (px2, py2), (0, 0, 255), 2)
                            
                            # اگر متن پلاک خوانده شده، نمایش آن
                            if plate.get('parsed') and plate['parsed'].get('formatted'):
                                plate_text = plate['parsed']['formatted']
                                # رسم متن پلاک
                                text_size, _ = cv2.getTextSize(
                                    plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )
                                cv2.rectangle(
                                    image,
                                    (px1, py2 + 5),
                                    (px1 + text_size[0] + 10, py2 + text_size[1] + 15),
                                    (0, 0, 255),
                                    -1,
                                )
                                cv2.putText(
                                    image, plate_text, (px1 + 5, py2 + text_size[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                )
                        
                        cv2.imwrite(processed_path, image)
                        
                        # ذخیره مسیر نسبی
                        relative_path = os.path.join(
                            'processed',
                            uploaded_image.uploaded_at.strftime('%Y/%m/%d'),
                            f'detected_{filename}'
                        )
                        uploaded_image.processed_image = relative_path
                    
                    uploaded_image.save()
                    
                    # نمایش پیام موفقیت
                    if result['has_vehicle']:
                        plate_msg = f" و {uploaded_image.plate_count} پلاک" if uploaded_image.has_plate else ""
                        messages.success(
                            request, 
                            f"✅ {result['vehicle_count']} خودرو{plate_msg} در تصویر شناسایی شد!"
                        )
                    else:
                        messages.info(request, "ℹ️ هیچ خودرویی در تصویر شناسایی نشد.")
                    
                    return redirect('result', pk=uploaded_image.pk)
                
            except Exception as e:
                messages.error(request, f"خطا در پردازش تصویر: {str(e)}")
                uploaded_image.delete()
    else:
        form = ImageUploadForm()
    
    # نمایش آخرین تصاویر
    recent_images = UploadedImage.objects.all()[:6]
    
    context = {
        'form': form,
        'recent_images': recent_images,
    }
    return render(request, 'detector/home.html', context)


def result(request, pk):
    """صفحه نمایش نتیجه تشخیص"""
    try:
        uploaded_image = UploadedImage.objects.get(pk=pk)
    except UploadedImage.DoesNotExist:
        messages.error(request, "تصویر مورد نظر یافت نشد.")
        return redirect('home')
    
    context = {
        'uploaded_image': uploaded_image,
    }
    return render(request, 'detector/result.html', context)


def history(request):
    """صفحه تاریخچه تصاویر"""
    images = UploadedImage.objects.all()
    
    context = {
        'images': images,
    }
    return render(request, 'detector/history.html', context)
