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
import os

PARENT_DIR = Path(__file__).resolve().parent.parent


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Iteration timed out")


# Set the timeout duration (in seconds)
timeout_duration = 5


class AssociationRuleMining:
    def __init__(self):
        """Makes associations between products from transactions"""

        self.orders_dict = dict()
        self.associations = dict()
        self.items = None

    def read_data(self, path=os.path.join(PARENT_DIR, "ml_models\mysql_data.json")):
        """
        path : "Path to Json Path "
        """
        df = pd.read_json(path)
        self.df = df
        return df

    def get_orders_dict(self, df):
        """Makes orders_dict by by grouping together items per order"""
        self.orders_dict = dict()
        uq_shops = df['shop_address'].unique()
        for i in list(uq_shops):
            temp_dict = dict()
            temp = df[df['shop_address'] == i]
            for index, row in temp.iterrows():
                if row['order_id'] not in temp_dict.keys():
                    temp_dict[row['order_id']] = [row['product_id']]
                else:
                    temp_dict[row['order_id']].append(row['product_id'])

            self.orders_dict[i] = temp_dict

        print("Successfully built orders dictionary")
        return self.orders_dict

    def get_associations(self, min_support=0.1, min_confidence=0.2, min_lift=2, min_length=2):

        """Uses apriori algorithm to make associations associations of items
        frequently bought together"""

        # signal.signal(signal.SIGALRM, timeout_handler)
        self.fetch_saved_products()
        for key, value in self.orders_dict.items():
            try:
                file_name = key.split(".")[0]
                file = open(os.path.join(PARENT_DIR, "ml_models/associations/") + file_name + ".json", 'w')

                # signal.alarm(5)
                print(key, "model training")
                temp_dict = dict()
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
                temp_dict[key] = rules
                print("file updated", file)
                json.dump(temp_dict, file)
                # signal.alarm(0)
            except TimeoutException:
                print("error duringfile updation", file)
                print(key, "timeout error")
                # Close the file
                file.close()

    # def fetch_saved_products(self, items_path=os.path.join(PARENT_DIR, "ml_models\items.json")):
    #     items = json.load(open(items_path, 'r'))
    #     self.items = items

    def fetch_saved_products(self, shop_address):
        items_path = os.path.join(PARENT_DIR, f"ml_models/orders_data/shops_products/{shop_address}.json")
        items = json.load(open(items_path, 'r'))
        self.items = items

    def get_prediction(self, store_name):
        file_name = store_name.split(".")[0]
        complete_name = os.path.join(PARENT_DIR, "ml_models/associations/{}.json").format(file_name)
        file_contents = open(complete_name, 'r').read()
        data = json.loads(file_contents)
        self.fetch_saved_products(store_name)
        data = [[self.items[store_name].get(str(item)) for item in record] for record in
                data[store_name]]
        results = data
        return results
