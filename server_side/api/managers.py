from django.contrib.auth.base_user import BaseUserManager
from django.utils.translation import gettext_lazy as _


class TRLVUserManager(BaseUserManager):
    """
    Custom user model manager where email is the unique identifiers
    for authentication instead of usernames.
    """
    def create_user(self, username_field, member_role, password=None, **other_fields):
        """
        Create and save a user with the given email and password.
        """
        if not email:
            raise ValueError(_("The Email must be set"))
        email = self.normalize_email(email)

        # instantiate the user model by passing the necessary values
        # for each of its attributes and now new attributes
        user = self.model(
            username_field=username_field, 
            member_role=member_role, 
            **other_fields)
        
        # don't pass password during instantiation but do so
        # when using the setter function for password
        user.set_password(password)

        # save the user to database
        user.save()
        
        # return new user
        return user

    def create_superuser(self, username_field, member_role, password=None, **other_fields):
        """
        Create and save a SuperUser with the given email and password.
        """
        other_fields.setdefault("is_staff", True)
        other_fields.setdefault("is_superuser", True)
        other_fields.setdefault("is_active", True)

        if not other_fields.get("is_staff"):
            raise ValueError(_("Superuser must have is_staff=True."))
        if not other_fields.get("is_superuser"):
            raise ValueError(_("Superuser must have is_superuser=True."))
        
        # use the create_user method as the final
        # layer for creating the super user
        return self.create_user(username_field, member_role, password, **other_fields)