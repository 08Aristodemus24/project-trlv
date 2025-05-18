from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from . import models, serializers

# Create your views here.
@api_view(["POST"])
def signup(request):
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

    # for getting primitive data
    for key in request.data.keys():
        print(f"key {key}: {request.POST.get(key)}")
    
    # for getting files
    for key in request.FILES.keys():
        print(f"key {key}: {request.FILES.get(key)}")