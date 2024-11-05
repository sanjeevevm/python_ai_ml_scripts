from django.shortcuts import render
from rest_framework.views import APIView
from data.utils import GenericAPIException
from rest_framework.response import Response
from rest_framework import status
# from data.recommendation_engine.getwise_content_filterieng import DataWrangling
from data.recommendation_engine.getwise_content_filtering_new import DataWrangling
# from data.recommendation_engine.getwise_association import AssociationRuleMining
from data.recommendation_engine.getwise_association_new import AssociationRuleMining
from data.recommendation_engine.getwise_product_recommendation import FetchingData
from data.serializers import ValidateArguments, ValidateProductTitle, RecommendProductSerializer

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
            obj.fetch_saved_model(shop_address)
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
        except FileNotFoundError:
            raise GenericAPIException(f'we dont have matching data related  {shop_address} ',
                                      status_code=404)
        except Exception as e:
            raise GenericAPIException(str(e))
        if not result:
            result = {'message': f"we dont have associated products related this {shop_address} "}
        return Response(data=result, status=status.HTTP_200_OK)


class RecommendProduct(APIView):
    serializer_class = RecommendProductSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        shop_address = data.get('shop_address')
        product_id = data.get('product_id')
        user_id = data.get('user_id')
        try:
            obj = FetchingData()
            model_path = os.path.join(PARENT_DIR, "ml_models/recommendation_api_models/{}.pth").format(
                shop_address.split('.')[0])
            model_result = obj.make_recommendations(user_id, int(product_id), model_path)
            product = model_result['product_id']
            if product:
                client = obj.building_connections()
                result = dict()
                result["user_id"] = user_id
                result['recommended_product'] = obj.get_product_info(client, shop_address, str(product))
                result['current_product'] = obj.get_product_info(client, shop_address, str(product_id))
            else:
                result = {'message': f"no product recommendend related this {product_id}"}
            status_code = status.HTTP_200_OK
        except GenericAPIException as generic_exception:
            result = {'message': str(generic_exception)}
            status_code = status.HTTP_400_BAD_REQUEST
        except Exception as e:
            result = {'message': str(e)}
            status_code = status.HTTP_400_BAD_REQUEST
        return Response(data=result, status=status_code)
