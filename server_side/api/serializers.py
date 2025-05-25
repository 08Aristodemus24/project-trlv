from rest_framework import serializers
from . import models
from django.contrib.auth.base_user import BaseUserManager

class TRLVUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.TRLVUser

        # If you only select id, name, username, and password in your 
        # serializer's Meta.fields: The API output when fetching user 
        # data would only include these four fields. When creating a 
        # new user (if this serializer is used for input), the serializer
        # would expect to receive data for name, username, and password 
        # (and id might be read-only or auto-generated). Fields like email, 
        # is_active, first_name, last_name, and date_created would not 
        # be part of this specific API representation.
        fields = [
            "id", 
            "username", 
            "password", 
            "email", 

            "first_name", 
            "last_name", 
            "mobile_no", 
            "member_role", 
            "bio", 
            "profile_img"
        ]

        # There is also a shortcut allowing you to specify arbitrary additional
        # keyword arguments on fields, using the extra_kwargs option. As in 
        # the case of read_only_fields, this means you do not need to explicitly 
        # declare the field on the serializer. If the field has already been 
        # explicitly declared on the serializer class, then the extra_kwargs 
        # option will be ignored.
        extra_kwargs = {
            "password": {
                # we set this to write only as obviously we dont want to
                # return a password that's visible and readable to us when we
                # we use a get request to retrieve user data
                "write_only": True
            },
        }

    # note we only use create assuming base user manager has not been
    # overridden and defined with new functionality like creating a user
    def create(self, validated_data):
        """
        assuming data has been validated by serializer this
        method is called in order to create a new user
        """
        
        # validated data is a dict that can be destructured
        # and split as their own individual keyword arg when passed
        # to the class or function e.g. 
        # User(username=validated_data["username"])
        username = validated_data.pop("username")
        member_role = validated_data.pop("member_role")
        password = validated_data.pop("password")
        
        # normalize email before instantiating model
        email = validated_data.pop("email")
        normed_email = BaseUserManager.normalize_email(email)

        # instantiate the user model by passing the necessary values
        # for each of its attributes and now new attributes
        user = self.Meta.model(
            username=username,
            member_role=member_role,
            email=normed_email,
            **validated_data)
        
        # don't pass password during instantiation but do so
        # when using the setter function for password, because 
        # if we just set password field to password, the 
        # password is not hashed and encrypted, as a result it 
        # will just be a plain string vulnerable to attack
        user.set_password(password)

        # save the user to database
        print("User {u} will now be created.".format(u=user))
        user.save()

        # all this can be done in one line using User.objects.create_user()
        # where wee pass our validated_data, no need to call save
        return user
