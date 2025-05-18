from rest_framework import serializers
from . import models

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
        fields = ["id", "username", "password", "email", "member_role"]

        # There is also a shortcut allowing you to specify arbitrary additional
        # keyword arguments on fields, using the extra_kwargs option. As in 
        # the case of read_only_fields, this means you do not need to explicitly 
        # declare the field on the serializer. If the field has already been 
        # explicitly declared on the serializer class, then the extra_kwargs 
        # option will be ignored.
        extra_kwargs = {
            "password": {
                # we set this to write only as obviously we dont want to
                # return a password that's visible and readable to us
                "write_only": True
            },
        }

    def create(self, validated_data):
        # validated data is a dict that can be destructured
        # and split as their own individual keyword arg when passed
        # to the class or function e.g. 
        # User(username=validated_data["username"])
        user = models.TRLVUser(**validated_data)

        # set password
        user.set_password(validated_data['password'])

        user.save()

        # all this can be done in one line using User.objects.create_user()
        # where wee pass our validated_data, no need to call save
        return user
