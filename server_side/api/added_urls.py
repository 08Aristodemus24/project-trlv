from django.urls import path

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from . import views


app_name = 'api'

urlpatterns = [
    path('signup', views.signup, name='signup'),

    # we do not need a login view as this will automatically handle
    # authentication assuming the credentials we submit in our request
    # also exists as a user in our database, as a result of signing up
    # this will return a refresh and access token 
    path('login', TokenObtainPairView.as_view(), name='login')
]