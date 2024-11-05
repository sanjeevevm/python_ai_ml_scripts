from rest_framework.routers import DefaultRouter, SimpleRouter
from django.conf import settings
from data.views import (SimilerProducts, AssociateProducts, RecommendProduct)
from django.urls import path, include

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('similar_products/', SimilerProducts.as_view(), name='similar_products'),
    path('associate_products/', AssociateProducts.as_view(), name='associate_products'),
    path('recommendend_product/', RecommendProduct.as_view(), name='recommendend_product'),
]
