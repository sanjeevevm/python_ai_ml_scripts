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
        self.df = None
        self.items = None
        self.similar_index = None

    def read_data(self, path):
        df = pd.read_json(path)
        self.df = df
        return df

    def get_products_by_shop(self, shop_address):
        res = self.df[self.df['shop_address'] == shop_address]
        res[['product_id', 'title']].drop_duplicates(inplace=True)
        res['clean_title'] = res['title'].apply(lambda x: self.clean_text(x))
        return res[['product_id', 'title', 'clean_title', 'price']]

    def clean_products_title(self, shop_address):
        res = self.df
        res.drop_duplicates(subset=['shop_address', 'product_id'], inplace=True)
        res['clean_title'] = res['title'].apply(lambda x: self.clean_text(x))
        res = res[res['clean_title'] != '']
        res = res[res['clean_title'].str.len() > 3]
        return res[
            ['shop_address', 'product_id', 'title', 'clean_title', 'price', 'product_line_items', 'ordurl', 'sku',
             'vendor', 'variant_title', 'line_price', 'image_url', 'product_type', 'product_handel']]

    def clean_text(self, text):
        if not text:
            return ""
        if not isinstance(text, str):
            return ""
        text = re.sub("\W", " ", text)
        text = text.lower()
        text = [token for token in text.split() if token.isalpha()]
        return " ".join(text)

    def get_contents_dict(self, df):
        shop_address = df['shop_address'].iloc[0]
        temp_dict = dict()
        for index, row in df.iterrows():
            temp_dict[row['product_id']] = {
                "product_title": row['clean_title'],
                "product_price": row['price'],
                "product_id": row['product_id'],
                "product_line_items": row['product_line_items'],
                "url": row['ordurl'], "sku": row['sku'], "vendor": row['vendor'],
                "variant_title": row['variant_title'], "line_price": row['line_price'],
                "image_url": row['image_url'],
                "product_type": None if pd.isna(row['product_type']) else row['product_type'],
                "product_handel": None if pd.isna(row['product_handel']) else row['product_handel']
            }
        return {shop_address: temp_dict}

    def vectorize_text(self, text_iter):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_vects = tfidf.fit_transform(text_iter)
        return tfidf_vects

    def get_text_sim(self, text_vectors):
        cosine_sim = linear_kernel(text_vectors, text_vectors)
        return cosine_sim

    def calculate_features(self, product_dict):
        similarity = dict()
        shop_address = list(product_dict.keys())[0]
        products_lst = np.array([product['product_title'] for product in product_dict[shop_address].values()])
        text_vects = self.vectorize_text(products_lst)
        text_similarity = self.get_text_sim(text_vects)
        similarity[shop_address] = text_similarity.tolist()
        return similarity

    def training(self):
        file_path = 'orders_distinct_shops.csv'
        df = pd.read_csv(file_path)
        shop_addresses = df['shop_address'].unique().tolist()
        if shop_addresses:
            print("Unique Shop Addresses:")
        for shop_address in shop_addresses:
            print(f"{shop_address} trainning start")
            items_path = os.path.join(PARENT_DIR, f"ml_models\orders_data\shops_products\{shop_address}.json")
            similarity_path = os.path.join(PARENT_DIR, f"ml_models\orders_data\shops_similar_products\{shop_address}.json")
            shop_data = self.read_data(os.path.join(PARENT_DIR, f"shop_orders_data/{shop_address}.json"))
            if len(shop_data) > 0:
                df_total = self.clean_products_title(shop_data)
                if len(df_total) > 0:
                    product_dict = self.get_contents_dict(df_total)
                    similarity = self.calculate_features(product_dict)

                    json.dump(product_dict, open(items_path, 'w'))
                    json.dump(similarity, open(similarity_path, 'w'))
                else:
                    print(f"{shop_address} has data less than 1 ")
                    continue
            else:
                print(f"{shop_address} has data less than 1 ")
                continue
        return self.items, self.similar_index

    def fetch_saved_model(self, shop_address):
        items_path = os.path.join(PARENT_DIR, f"ml_models\orders_data\shops_products\{shop_address}.json")
        similarity_path = os.path.join(PARENT_DIR, f"ml_models\orders_data\shops_similar_products\{shop_address}.json")

        try:
            with open(items_path, 'r') as items_file, open(similarity_path, 'r') as similarity_file:
                items = json.load(items_file)
                similarity = json.load(similarity_file)
                self.items = items
                self.similar_index = similarity
        except FileNotFoundError:
            raise FileNotFoundError(f"Data files for shop address {shop_address} not found.")

    def get_most_similar_text(self, store, product_name, n=3):
        """Returns n most similar products, based on item-name text"""
        product = list(self.items[store].keys())
        try:
            idx = product.index(product_name)
        except ValueError:
            raise ValueError(f"Product {product_name} not found in the list for store {store}")

        sim_scores = self.similar_index[store][idx]
        closest_matches_ind = np.argsort(sim_scores)[::-1][:n + 1]
        closest_matches_names = dict()

        closest_matches_names['current_product'] = self.items[store][product[closest_matches_ind[0]]]
        closest_matches_names['similar_products'] = [self.items[store][product[i]] for i in closest_matches_ind][1:]

        return closest_matches_names
