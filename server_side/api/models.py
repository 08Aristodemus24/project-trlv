from django.db import models
from django.contrib.auth.models import User, AbstractUser , PermissionsMixin
from .managers import TRLVUserManager

# Create your models here.
class TRLVUser(AbstractUser):
    """
    by defalt django will have the following fields that are supplied
    through the User class' arguments e.g. id, password, is_superuser,
    username, first_name, last_name, email, etc. which have certain data
    types that we must match during input

    <ManyToOneRel: admin.logentry>, 
    <django.db.models.fields.AutoField: id>, 
    <django.db.models.fields.CharField: password>, 
    <django.db.models.fields.DateTimeField: last_login>, 
    <django.db.models.fields.BooleanField: is_superuser>, 
    <django.db.models.fields.CharField: username>, 
    <django.db.models.fields.CharField: first_name>, 
    <django.db.models.fields.CharField: last_name>, 
    <django.db.models.fields.EmailField: email>, 
    <django.db.models.fields.BooleanField: is_staff>, 
    <django.db.models.fields.BooleanField: is_active>, 
    <django.db.models.fields.DateTimeField: date_joined>, 
    <django.db.models.fields.related.ManyToManyField: groups>, 
    <django.db.models.fields.related.ManyToManyField: user_permissions>

    since we have an extra field called member role we would have to create
    this field explicitly
    """

    # new fields for a user will be a member role and profile image
    member_role = models.CharField(verbose_name="Member Role", default="Elder", max_length=50)
    profile_img = models.ImageField(verbose_name="Profile Image", width_field=100, height_field=100, )
    
    REQUIRED_FIELDS = ["member_role"]

    # by defautl django will require you to enter the username field
    # to let us have a unique identifier
    # username = None
    objects = TRLVUserManager()

    def __str__(self):
        return self.get_username()


# def test(key: str=...):
#     pass