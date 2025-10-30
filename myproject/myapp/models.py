from django.db import models
from django.utils.translation import gettext as _
# Create your models here.
class ITEM(models.Model):
    professor_name = models.CharField(_("professor_name"),max_length=150)
    school_name = models.CharField(_("school_name"),max_length=150)
    department_name = models.CharField(_("department_name"),max_length=150)
    star_rating = models.FloatField(_("star_rating"))
    course = models.CharField(_("name_not_onlines"),max_length=150)
    difficulty = models.IntegerField(_("student_difficult"))
    would_take_agains = models.BooleanField(_("would_take_agains"),default=False)
    help_useful = models.IntegerField(_("help_useful"))
    comments = models.CharField(_("comments"),max_length=255)

    class Meta:
        db_table = "ITEM"


   
