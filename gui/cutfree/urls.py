from django.urls import path
from cutfree.views import home

urlpatterns = [
    path('', home, name="home"),
]