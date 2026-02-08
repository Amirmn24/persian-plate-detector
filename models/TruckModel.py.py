from django.db import models

class TruckInfo(models.Model):
    """مدل اطلاعات کامیون"""

    VEHICLE_TYPE_CHOICES = [
        ('trailer_dump', 'تریلی کمپرسی'),
        ('trailer_non_dump', 'تریلی غیر کمپرسی'),
        ('trailer_flatbed', 'تریلی کفس'),
        ('single_axle', 'تک'),
        ('tandem', 'جفت'),
        ('khavar', 'خاور'),
        ('nissan', 'نیسان'),
        ('pickup', 'وانت'),
        ('other', 'سایر'),
    ]

    LOADING_TYPE_CHOICES = [
        ('open', 'باز'),
        ('closed', 'بسته'),
        ('refrigerated', 'یخچال دار'),
        ('tank', 'تانکر'),
        ('flatbed', 'کفی'),
        ('other', 'سایر'),
    ]
    
    plate_number = models.CharField(
        max_length=20,
        verbose_name='پلاک',
        null=False,
        blank=False
    )
    
    vehicle_type = models.CharField(
        max_length=20,
        choices=VEHICLE_TYPE_CHOICES,
        verbose_name='نوع ماشین',
        null=True,
        blank=True
    )
    
    is_smart_truck = models.BooleanField(
        default=False,
        verbose_name='هوشمند کامیون',
        null=True,
        blank=True
    )
    
    manufacturer = models.CharField(
        max_length=100,
        verbose_name='کارخانه کامیون',
        null=True,
        blank=True
    )
    
    loading_type = models.CharField(
        max_length=20,
        choices=LOADING_TYPE_CHOICES,
        verbose_name='نوع بارگیر',
        null=True,
        blank=True
    )

    class Meta:
        verbose_name = 'حمل | اطلاعات کامیون'
        verbose_name_plural = 'حمل | اطلاعات کامیون‌ها'
        ordering = ['transport', 'plate_number']

    def __str__(self):
        return f"اطلاعات کامیون {self.plate_number or 'بدون پلاک'}"

