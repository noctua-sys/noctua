from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('Administration/', include('Administration.urls')),
    # path('Administration2/', include('Administration.urls')),
    # path('Administration3/', include('Administration.urls')),
]
