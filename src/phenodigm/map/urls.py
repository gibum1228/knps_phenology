from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analysis/', views.analysis, name='analysis')
    # path('analysis/<name>/', views.analysis, name='analysis') # name은 동적인 값이 들어감
]