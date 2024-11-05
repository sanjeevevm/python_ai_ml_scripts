import plotly.express as px
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import OrderedDict
import json
import os
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent


class DataWrangling:
    def __init__(self):
        """
        - Initial Constructor
        """
        self.df = None
        self.uq_shops = None
        self.items = None
        self.similar_index = None

    def read_data(self, path=os.path.join(PARENT_DIR, "ml_models\mysql_data.json")):
        """
        path : "Path to Json Path "
        """
        # path = os.path.join(PARENT_DIR, "ml_models\mysql_data.json")
        df = pd.read_json(path)
        self.df = df
        return df

    def unique_shops(self):
        """
        Getting Unique Shops
        """
        self.uq_shops = list(self.df['shop_address'].unique())

    def get_products_by_shop(self, shop_address):
        """
        fetching products_name
        """
        res = self.df[self.df['shop_address'] == shop_address]
        res[['product_id', 'title']].drop_duplicates(inplace=True)
        res['clean_title'] = res['title'].apply(lambda x: self.clean_text(x))

        return res[['product_id', 'title', 'clean_title', 'price']]

    def clean_products_title(self):
        res = self.df
        res.drop_duplicates(subset=['shop_address', 'product_id'], inplace=True)
        res['clean_title'] = res['title'].apply(lambda x: self.clean_text(x))
        res = res[res['clean_title'] != '']
        res = res[res['clean_title'].str.len() > 3]
        return res[
            ['shop_address', 'product_id', 'title', 'clean_title', 'price', 'product_line_items', 'url', 'sku',
             'vendor', 'variant_title', 'line_price', 'image_url', 'product_type', 'product_handel']]

    def clean_text(self, text):
        """Cleans input text by removing non-words and numbers"""

        if not text:
            return ""
        text = re.sub("\W", " ", text)
        text = text.lower()
        text = [token for token in text.split() if token.isalpha()]
        return " ".join(text)

    def get_contents_dict(self, df):
        total_shops = list(df['shop_address'].unique())
        products = dict()
        for i in total_shops:
            temp = df[df['shop_address'] == i]
            temp_dict = OrderedDict()
            for index, row in temp.iterrows():
                temp_dict[row['product_id']] = {"product_title": row['clean_title'],
                                                "product_price": row['price'],
                                                "product_id": row['product_id'],
                                                "product_line_items": row['product_line_items'],
                                                "url": row['url'], "sku": row['sku'], "vendor": row['vendor'],
                                                "variant_title": row['variant_title'], "line_price": row['line_price'],
                                                "image_url": row['image_url'],
                                                "product_type": row['product_type'],
                                                "product_handel": row['product_handel']
                                                }

                products[i] = temp_dict
                ordered_dict = OrderedDict(products)
                self.items = ordered_dict

        return products

    def vectorize_text(self, text_iter):
        """Given a text iterable, returns TF-IDF scoring vectors for it"""
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_vects = tfidf.fit_transform(text_iter)

        return tfidf_vects

    def get_text_sim(self, text_vectors):
        """Return pairwise cosine similarity on provided vectors"""

        cosine_sim = linear_kernel(text_vectors, text_vectors)

        return cosine_sim

    def calculate_features(self, product_dicts):
        similarity = dict()
        for i in list(product_dicts.keys()):
            products_lst = np.array([product['product_title'] for product in product_dicts[i].values()])
            text_vects = self.vectorize_text(products_lst)
            text_similarity = self.get_text_sim(text_vects)
            similarity[i] = text_similarity.tolist()
        ordered_dict = OrderedDict(similarity)

        self.similar_index = ordered_dict
        return similarity

    def training(self, items_path=os.path.join(PARENT_DIR, "ml_models\items.json"),
                 similarity_path=os.path.join(PARENT_DIR, "ml_models\similar.json")):
        df = self.read_data()
        df_total = self.clean_products_title()
        product_dicts = self.get_contents_dict(df_total)
        similarity = self.calculate_features(product_dicts)

        json.dump(self.items, open(items_path, 'w'))
        json.dump(self.similar_index, open(similarity_path, 'w'))
        return self.items, self.similar_index

    def fetch_saved_model(self, items_path=os.path.join(PARENT_DIR, "ml_models\items.json"),
                          similarity_path=os.path.join(PARENT_DIR, "ml_models\similar.json")):
        items = json.load(open(items_path, 'r'))
        similarity = json.load(open(similarity_path, 'r'))
        self.items = items
        self.similar_index = similarity

    def get_most_similar_text(self, store, product_name, n=3):

        """Returns n most similar products, based on item-name text"""
        product = list(self.items[store].keys())
        idx = product.index(product_name)

        sim_scores = self.similar_index[store][idx]
        closest_matches_ind = np.argsort(sim_scores)[::-1][:n + 1]
        closest_matches_names = dict()

        closest_matches_names['current_product'] = self.items[store][product[closest_matches_ind[0]]]
        closest_matches_names['similar_products'] = [self.items[store][product[i]] for i in closest_matches_ind][1:]

        return closest_matches_names

    # def get_most_similar_text(self, product_name, n=3):
    #     """Returns n most similar products, based on item-name text"""
    #     product = list(self.items[self.shop_address].keys())
    #     idx = product.index(product_name)
    #
    #     sim_scores = self.similar_index[self.shop_address][idx]
    #     closest_matches_ind = np.argsort(sim_scores)[::-1][:n + 1]
    #     closest_matches_names = dict()
    #
    #     closest_matches_names['current_product'] = self.items[self.shop_address][product[closest_matches_ind[0]]]
    #     closest_matches_names['similar_products'] = [self.items[self.shop_address][product[i]] for i in
    #                                                  closest_matches_ind][1:]
    #
    #     return closest_matches_names

    # shop_address = "leprosy-mission.myshopify.com"
    # shop_data_wrangling = ShopDataWrangling(shop_address)
    # shop_data_wrangling.read_data(f"shop_orders_data/{shop_address}.json")
    # # Call other methods as needed
    # result = shop_data_wrangling.get_most_similar_text(product_name="6792406270035", n=3)
    # print(result)
