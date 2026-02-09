from django.db import models
from django.utils import timezone


class UploadedImage(models.Model):
    """مدل برای ذخیره تصاویر آپلود شده"""
    
    image = models.ImageField(
        upload_to='uploads/%Y/%m/%d/',
        verbose_name='تصویر'
    )
    
    uploaded_at = models.DateTimeField(
        default=timezone.now,
        verbose_name='تاریخ آپلود'
    )
    
    has_vehicle = models.BooleanField(
        default=False,
        verbose_name='دارای خودرو'
    )
    
    vehicle_count = models.IntegerField(
        default=0,
        verbose_name='تعداد خودرو'
    )
    
    detection_result = models.JSONField(
        null=True,
        blank=True,
        verbose_name='نتیجه تشخیص'
    )
    
    processed_image = models.ImageField(
        upload_to='processed/%Y/%m/%d/',
        null=True,
        blank=True,
        verbose_name='تصویر پردازش شده'
    )
    
    # فیلدهای مربوط به پلاک
    has_plate = models.BooleanField(
        default=False,
        verbose_name='دارای پلاک'
    )
    
    plate_count = models.IntegerField(
        default=0,
        verbose_name='تعداد پلاک'
    )
    
    plate_data = models.JSONField(
        null=True,
        blank=True,
        verbose_name='اطلاعات پلاک'
    )
    
    class Meta:
        verbose_name = 'تصویر آپلود شده'
        verbose_name_plural = 'تصاویر آپلود شده'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"تصویر {self.id} - {self.uploaded_at.strftime('%Y/%m/%d %H:%M')}"
