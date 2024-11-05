from rest_framework import serializers


class ValidateArguments(serializers.Serializer):
    shop_address = serializers.CharField(max_length=100, required=True)
    product_id = serializers.CharField(max_length=100, required=True, allow_null=False)
    num_of_products = serializers.IntegerField(min_value=1, max_value=15, required=False)


class ValidateProductTitle(serializers.Serializer):
    shop_address = serializers.CharField(max_length=100, required=True)


class RecommendProductSerializer(serializers.Serializer):
    shop_address = serializers.CharField(max_length=100, required=True)
    product_id = serializers.CharField(max_length=100, required=True, allow_null=False)
    user_id = serializers.CharField(max_length=100, required=True, allow_null=False)
