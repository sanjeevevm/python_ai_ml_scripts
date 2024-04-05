from threading import Thread
import functools
import plotly.express as px
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from apyori import apriori
import json
import signal
from pathlib import Path
import itertools
import os

PARENT_DIR = Path(__file__).resolve().parent.parent


def timeout(timeout, attribute_name):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] for shop_address [%s shop_address] timeout [%s seconds] exceeded!' % (
                func.__name__, getattr(args[0], attribute_name), timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


class AssociationRuleMining:
    def __init__(self):
        """Makes associations between products from transactions"""
        self.orders_dict = dict()
        self.associations = dict()
        self.shop_address = None

    # def read_data(self, path='../data_formation/mysql_data.json'):
    def read_data(self, path=os.path.join(PARENT_DIR, "ml_models\mysql_data.json")):
        """
        path : "Path to Json Path "
        """
        df = pd.read_json(path)
        self.df = df
        return df

    def get_orders_dict(self, df):
        """Makes orders_dict by grouping together items per order"""
        self.orders_dict = dict()

        if df.empty:
            print("Empty dataframe. Cannot build orders dictionary.")
            # raise ValueError("Empty dataframe. Cannot build orders dictionary.")
            return self.orders_dict

        # If there is only one shop in the dataframe, process it directly
        if len(df['shop_address'].unique()) == 1:
            shop_address = df['shop_address'].iloc[0]
            self.shop_address = shop_address
            # print(shop_address)
            temp_dict = dict()

            for index, row in df.iterrows():
                if row['order_id'] not in temp_dict:
                    temp_dict[row['order_id']] = [row['product_id']]
                else:
                    temp_dict[row['order_id']].append(row['product_id'])

            self.orders_dict[shop_address] = temp_dict
            # print(self.orders_dict)
        else:
            # If multiple shops are present, process each shop separately
            uq_shops = df['shop_address'].unique()
            for i in list(uq_shops):
                temp_dict = dict()
                temp = df[df['shop_address'] == i]
                for index, row in temp.iterrows():
                    if row['order_id'] not in temp_dict:
                        temp_dict[row['order_id']] = [row['product_id']]
                    else:
                        temp_dict[row['order_id']].append(row['product_id'])

                self.orders_dict[i] = temp_dict

        print(f"Successfully built orders dictionary for {self.shop_address}")
        return self.orders_dict

    @timeout(5, attribute_name="shop_address")
    def get_associations(self, min_support=0.1, min_confidence=0.2, min_lift=2, min_length=2):
        # """Uses apriori algorithm to make associations of items frequently bought together"""
        # signal.signal(signal.SIGALRM, timeout_handler)
        # SIGALRM = getattr(signal, 'SIGALRM', None)
        # signal.signal(SIGALRM, timeout_handler)

        for key, value in self.orders_dict.items():
            try:
                file_name = key.split(".")[0]
                # file = open("../data_formation/associations/" + file_name + ".json", 'w')
                file = open(os.path.join(PARENT_DIR, "ml_models/associations_new/") + file_name + ".json", 'w')

                total_ids = list(value.values())
                transactions = [transaction for transaction in total_ids if len(transaction) >= 3]
                transactions = [list(set(transaction)) for transaction in transactions]
                total_lists = transactions if len(transactions) > 1 else total_ids

                if len(total_lists) > 50:
                    temp_val = list(apriori(total_ids[:10], min_support=1, min_confidence=1,
                                            min_lift=min_lift, min_length=min_length))
                else:
                    temp_val = list(apriori(total_ids[:10], min_support=min_support, min_confidence=min_confidence,
                                            min_lift=min_lift, min_length=min_length))
                rules = [tuple(ass[0]) for ass in temp_val]
                temp_dict = {key: rules}
                json.dump(temp_dict, file)

            except Exception as e:
                print(f"{key} - {e}")
                raise TimeoutError(f"{key} - Timeout Error")
            file.close()

    # def get_prediction(self, store_name):
    #     file_name = store_name.split(".")[0]
    #     complete_name = "../data_formation/associations/{}.json".format(file_name)
    #     if os.path.exists(complete_name):
    #         file_contents = open(complete_name, 'r').read()
    #         data = json.loads(file_contents)
    #         results = data[store_name]
    #         return results
    #     else:
    #         raise FileNotFoundError(f"File for store {store_name} not found")

    def get_product_associations(self, shop_address):
        """Returns associated products against slug"""

        if self.associations:
            product_associations = [set(ass) - {slug} for ass in self.associations[shop_address] if slug in ass]
            product_associations = set(itertools.chain.from_iterable(product_associations))

            return list(product_associations)
        else:
            raise ValueError("Product associations need to be built or loaded first")

    def process_all_shops(self, file_path):
        """
        Iterate through all shop names, read each shop's data, and perform operations.
        """
        df = pd.read_csv(file_path)
        shop_addresses = df['shop_address'].unique().tolist()
        # index_to_start = shop_addresses.index('wheelhousecycleclub.myshopify.com')

        for shop_address in shop_addresses:
            self.process_single_shop(shop_address)
        # for shop_address in shop_addresses[index_to_start:]:
        # for shop_address in ['7fc8d0-3.myshopify.com']:

    def process_single_shop(self, shop_address):
        """
        Read data for a single shop and perform operations.
        """
        association_miner = AssociationRuleMining()

        shop_path = os.path.join(PARENT_DIR, f"shop_orders_data/{shop_address}.json")
        self.read_data(shop_path)
        # print(shop_path)
        data_frame = association_miner.read_data(shop_path)
        orders_dictionary = association_miner.get_orders_dict(data_frame)
        # print(orders_dictionary)

        try:
            association_miner.get_associations()

        except Exception as e:
            print(f"{e}")

    def fetch_saved_products(self, shop_address):
        items_path = os.path.join(PARENT_DIR, f"ml_models/orders_data/shops_products/{shop_address}.json")
        items = json.load(open(items_path, 'r'))
        self.items = items

    def get_prediction(self, store_name):
        file_name = store_name.split(".")[0]
        complete_name = os.path.join(PARENT_DIR, "ml_models/associations_new/{}.json").format(file_name)
        if os.path.exists(complete_name):
            file_contents = open(complete_name, 'r').read()
            data = json.loads(file_contents)
            self.fetch_saved_products(store_name)
            data = [[self.items[store_name].get(str(item)) for item in record] for record in
                    data[store_name]]
            results = data
            return results
        else:
            raise FileNotFoundError(f"File for store {store_name} not found")

# association_miner = AssociationRuleMining()
# data_frame = association_miner.read_data()
# orders_dictionary = association_miner.get_orders_dict(data_frame)
# association_miner.get_associations()
# prediction_result = association_miner.get_prediction("saeed-malas.myshopify.com")
# print(prediction_result)

# file_path = './orders_distinct_shops.csv'
# association_miner = AssociationRuleMining()
# association_miner.process_all_shops(file_path)
