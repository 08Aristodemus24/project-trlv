from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from . import models, serializers

# Create your views here.
@api_view(["POST"])
def signup(request):
    """
    creates a new user with given credentials and other
    information
    """
    # Larry Miguel
    # Cueva
    # BladeRunne
    # MichaelAveuc571@gmail.com
    # 097-745-1021
    # +63
    # Elder
    # adsfa
    # null
    # print(request.data)
    # print(request.files)

    # # for getting primitive data
    # for key in request.data.keys():
    #     print(f"key {key}: {request.POST.get(key)}")
    
    # # for getting files
    # for key in request.FILES.keys():
    #     print(f"key {key}: {request.FILES.get(key)}")

    data = {
        # fields string below must not be empty
        "username": request.data.get("user_name") or None,
        "password": request.data.get("password") or None,
        "email": request.data.get("email_address") or None,

        # fields strings below can be empty
        "first_name": request.data.get("first_name"),
        "last_name": request.data.get("last_name"),
        "mobile_no": f'({request.data.get("country_code")}) {request.data.get("mobile_num")}',
        "member_role": request.data.get("member_role"),
        "bio": request.data.get("bio"),
        "profile_img": request.FILES.get("profile_image"),
    }

    user_serializer = serializers.TRLVUserSerializer(data=data)
    
    if user_serializer.is_valid():
        print("Submitted credentials are valid. Commencing user creation...")
        # valid_data = user_serializer.validated_data
        # print(valid_data)

        # call method save as it implicitly calls create
        # which we know we have overridden with our own logic
        # which will then create a user
        user_serializer.save()

        # when we test it in the shell we see that we have created
        # the user object
        # >>> from api import models
        # >>>
        # >>> models.TRLVUser.objects.all()
        # <QuerySet [<TRLVUser: BladeRunne>]>
        # >>> users = models.TRLVUser.objects.all()
        # >>> user = users.get(username="BladeRunne")
        # >>>
        # >>> user
        # <TRLVUser: BladeRunne>
        return Response(user_serializer.data)

    else:
        print("Submitted credentials invalid. Following errors occured")
        print(user_serializer.errors)
        return Response(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
