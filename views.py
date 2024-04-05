from django.shortcuts import render
from rest_framework.views import APIView
from data.utils import GenericAPIException
from rest_framework.response import Response
from rest_framework import status
from data.recommendation_engine.getwise_content_filterieng import DataWrangling
from data.recommendation_engine.getwise_association import AssociationRuleMining
from data.serializers import ValidateArguments, ValidateProductTitle

import os
import pandas as pd
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent


# Create your views here.
class SimilerProducts(APIView):
    serializer_class = ValidateArguments

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        shop_address = data.get('shop_address')
        num_of_products = data.get('num_of_products', 3)
        product_id = data.get('product_id')
        try:
            obj = DataWrangling()
            obj.fetch_saved_model()
            result = obj.get_most_similar_text(shop_address, product_id, num_of_products)
        except KeyError:
            raise GenericAPIException(f'we dont have matching data related {product_id} or {shop_address} ',
                                      status_code=404)
        except Exception as e:
            return Response(f'we dont have matching data related {product_id} or {shop_address}',
                            status=status.HTTP_400_BAD_REQUEST)
        if not result:
            result = {'message': f"similar products related this {product_id} doesn't exist "}
        return Response(data=result, status=status.HTTP_200_OK)


class AssociateProducts(APIView):
    serializer_class = ValidateProductTitle

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        shop_address = data.get('shop_address')
        try:
            obj = AssociationRuleMining()
            result = obj.get_prediction(shop_address)
        except KeyError:
            raise GenericAPIException(f'we dont have matching data related  {shop_address} ',
                                      status_code=404)
        except Exception as e:
            raise Exception(str(e))
        if not result:
            result = {'message': f"similar products related this {shop_address} doesn't exist "}
        return Response(data=result, status=status.HTTP_200_OK)
