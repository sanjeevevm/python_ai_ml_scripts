import pandas as pd
from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import pdb

import pymongo
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
from datetime import datetime
from bson import BSON
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta
from data.utils import GenericAPIException


class FetchingData():
    def building_connections(self, ):
        # SSH tunnel configuration
        ssh_host = '44.212.21.160'
        ssh_port = 22332
        ssh_user = 'mongdb'
        ssh_password = 'uc2EvW7xzcizCA8Bf219'

        # MongoDB connection details
        mongo_user = 'wiseraidb'
        mongo_password = 'B5cN5A19T'
        mongo_database = 'wiseraidb'  # Replace with your actual database name
        mongo_collection = 'wiser_evmaidata_clicks'  # Replace with your actual collection name

        # Establish SSH tunnel
        tunnel = SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_password,
            remote_bind_address=('localhost', 27017)
        )
        tunnel.start()

        # Set up MongoDB client
        uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:{tunnel.local_bind_port}/{mongo_database}"
        client = MongoClient(uri)
        return client

    def fetching_unique_shops(self, file_name):
        df = pd.read_csv(file_name)
        total_shop_address = list(df['shop_address'].unique())
        return total_shop_address

    def fetching_unique_shops_db(self, client):
        db = client["wiseraidb"]
        collection = db["wiser_evmaidata_clicks"]
        query = {
            '$and': [
                {'product_id': {'$ne': None}},
                {'product_id': {'$ne': 'undefined'}},
                {'iplog': {'$ne': None}},
                {'iplog': {'$ne': 'undefined'}}
            ]
        }
        unique_shops = collection.distinct('shop', query)
        return unique_shops

    def fetching_data(self, client, shop_name):
        filter_val = {
            'shop': '{}'.format(shop_name),
            '$and': [{'product_id': {'$ne': None}}, {'product_id': {'$ne': 'undefined'}}],
            '$and': [{'iplog': {'$ne': None}}, {'iplog': {'$ne': 'undefined'}}],
        }
        project = {
            'shop': 1,
            'product_id': 1,
            'iplog': 1,
            'created_date': 1
        }

        limit = 10000
        # print(filter_val)
        result = client['wiseraidb']['wiser_evmaidata_clicks'].find(
            filter=filter_val,
            limit=limit,
            projection=project
        )
        return result

    def fetch_and_save_shop_data(unique_shops="./unique_shops.csv"):

        # Read shop names from the input file
        with open(input_file, "r") as file:
            unique_shops = [line.strip() for line in file.readlines()]

        for shop_name in unique_shops:
            cursor = obj.fetching_data(client, shop_name)
            file_name = shop_name.split(".")[0]
            pd.DataFrame(cursor).to_csv("./shop_data/" + file_name + ".json")

    def fetching_data_unique(self, client, shop_name, start_date, end_date):
        filter_val = {
            'shop': '{}'.format(shop_name),
            '$and': [{'product_id': {'$ne': None}}, {'product_id': {'$ne': 'undefined'}}],
            '$and': [{'iplog': {'$ne': None}}, {'iplog': {'$ne': 'undefined'}}],
            'created_date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }

        project = {
            '_id': 0,
            'shop': 1,
            'product_id': 1,
            'iplog': 1,
            'cdate': 1
        }

        # pipeline = [
        #     {'$match': filter_val},
        #     {'$project': project},
        #     {'$group': {'_id': '$created_date', 'doc': {'$first': '$$ROOT'}}},
        #     {'$replaceRoot': {'newRoot': '$doc'}},
        #     {'$limit': 5}
        # ]
        pipeline = [
            {'$match': filter_val},
            {'$project': project},
            {'$limit': 300000}
        ]

        result = client['wiseraidb']['wiser_evmaidata_clicks'].aggregate(pipeline)

        return result

    def fetch_max_dates(self, client):
        max_date_pipeline = [
            {'$group': {'_id': '$shop', 'max_date': {'$max': '$cdate'},
                        'min_date': {'$min': '$cdate'}}}
        ]

        max_dates = client['wiseraidb']['wiser_evmaidata_clicks'].aggregate(max_date_pipeline)
        return max_dates

    def get_shops_data(self, client):
        shops_max_dates = obj.fetch_max_dates(client)
        df = pd.DataFrame(shops_max_dates)
        df.rename(columns={'_id': 'shop_address'}, inplace=True)
        df.dropna(subset=['shop_address'], inplace=True)
        df.to_csv("./unique_shops_now.csv")
        return df

    def fetching_specific_user_data(self, client, shop_name, user_id):
        """
        """
        filter_val = {
            'shop': '{}'.format(shop_name),

            'iplog': ''.format(user_id)
        }

        project = {
            'shop': 1,
            'product_id': 1,
            'iplog': 1,
            'cdate': 1
        }

        limit = 500
        result = client['wiseraidb']['wiser_evmaidata_clicks'].find(
            filter=filter_val,
            limit=limit,
            projection=project
        )
        return result

    def shops_latest_month_data(self, client, shops_data):
        for index, row in shops_data.iterrows():
            if pd.notna(row['shop_address']):
                # if row['shop_address'] == 'meinekette-de.myshopify.com':
                # Convert the ending date to datetime
                end_date = pd.to_datetime(row['max_date']).replace(hour=0, minute=0, second=0)
                # Calculate the start date by subtracting 30 days
                start_date = (end_date - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
                end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
                shop_name = row['shop_address']
                results = obj.fetching_data_unique(client, shop_name, start_date, end_date)
                data = list(results)
                # print(f"{shop_name}:{len(data)}")
                file_path = "./shop_data/{}.json".format(shop_name.split('.')[0])
                pd.DataFrame(data).to_json(file_path, orient="records")

    def train_shop_data(self, shops_data):
        for index, row in shops_data.iterrows():
            if pd.notna(row['shop_address']):
                if row['shop_address'] == 'meinekette-de.myshopify.com':
                    data_wrangler = DataWrangling()
                    # data_df = data_wrangler.read_specific_shop('./shop_data/meinekette-de3.json')
                    file_path = "./shop_data/{}.json".format(row['shop_address'].split('.')[0])
                    # print(file_path)
                    data_df = data_wrangler.read_specific_shop(file_path)
                    user_to_index = {user_id: i for i, user_id in enumerate(data_df['iplog'].unique())}
                    item_to_index = {item_id: i for i, item_id in enumerate(data_df['product_id'].unique())}

                    train_df, test_df = data_wrangler.train_test_split_df(data_df)
                    train_df, test_df, min_date, max_date = data_wrangler.scaling_date(train_df, test_df)
                    train_df, test_df = data_wrangler.user_item_matrix(data_df, train_df, test_df)

                    train_df = train_df.sort_values(['user_index', 'scaled_created_date'])
                    # Shift the 'item_index' to get the next purchased product
                    train_df['next_item_index'] = train_df.groupby('user_index')['item_index'].shift(-1)
                    # Drop the last rows for each user since there's no "next" item
                    train_df = train_df.dropna()
                    train_df['next_item_index'] = train_df['next_item_index'].astype(int)

                    test_df = test_df.sort_values(['user_index', 'scaled_created_date'])
                    # Shift the 'item_index' to get the next purchased product
                    test_df['next_item_index'] = test_df.groupby('user_index')['item_index'].shift(-1)
                    # Drop the last rows for each user since there's no "next" item
                    test_df = test_df.dropna()
                    test_df['next_item_index'] = test_df['next_item_index'].astype(int)

                    train_user_indices = torch.tensor(train_df['user_index'].values, dtype=torch.long)
                    train_item_indices = torch.tensor(train_df['item_index'].values, dtype=torch.long)
                    train_click_dates = torch.tensor(train_df['scaled_created_date'].values, dtype=torch.float32)
                    # train_products = torch.tensor(train_df['item_index'].values, dtype=torch.torch.float32)  # Replace 'your_target_column' with your actual target column
                    # Extract the 'next_item_index' as the target variable
                    # train_products = torch.tensor(train_df['next_item_index'].values, dtype=torch.float32)
                    train_products = torch.tensor(train_df['next_item_index'].values, dtype=torch.long)

                    test_user_indices = torch.tensor(test_df['user_index'].values, dtype=torch.long)
                    test_item_indices = torch.tensor(test_df['item_index'].values, dtype=torch.long)
                    test_click_dates = torch.tensor(test_df['scaled_created_date'].values, dtype=torch.float32)
                    # test_products = torch.tensor(test_df['item_index'].values, dtype=torch.float32)
                    # test_products = torch.tensor(test_df['next_item_index'].values, dtype=torch.float32)
                    test_products = torch.tensor(test_df['next_item_index'].values, dtype=torch.long)

                    train_dataset = TensorDataset(train_user_indices, train_item_indices, train_click_dates,
                                                  train_products)
                    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

                    test_dataset = TensorDataset(test_user_indices, test_item_indices, test_click_dates, test_products)
                    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

                    # num_classes = len(train_df['next_item_index'].unique())
                    num_users = len(data_df['iplog'].unique())
                    num_items = len(data_df['product_id'].unique())
                    # num_classes = len(train_df['next_item_index'].unique()) +1
                    num_classes = train_df['next_item_index'].max() + 1
                    embedding_size = 32
                    model = RecommenderNet(num_users, num_items, embedding_size, num_classes)

                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                    # Training loop
                    num_epochs = 40
                    for epoch in range(num_epochs):
                        model.train()
                        running_loss = 0.0
                        for batch in train_loader:
                            user_indices, item_indices, click_dates, train_labels = batch  # Move data to GPU

                            click_dates = click_dates.unsqueeze(1)

                            optimizer.zero_grad()
                            outputs = model(user_indices, item_indices, click_dates)
                            loss = criterion(outputs, train_labels)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()

                        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

                    print(f"Training complete for:{row['shop_address']}")

                    model.eval()
                    model_save_path = "./train_models/{}.pth".format(row['shop_address'].split('.')[0])
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'num_users': num_users,
                        'num_items': num_items,
                        'embedding_size': embedding_size,
                        'num_classes': num_classes,
                        'user_to_index': user_to_index,
                        'item_to_index': item_to_index,
                        'min_date': min_date,
                        'max_date': max_date
                    }
                    torch.save(checkpoint, model_save_path)

    def get_product_info(self, client, shop_address, product_id):
        # Define the query
        query = {"product_id": product_id, "shop": shop_address}
        # Define the projection to include only specific fields
        projection = {"product_id": 1, "product_handel": 1, "product_type": 1, "product_title": 1, "product_url": 1,
                      "_id": 0}
        # result = client['wiseraidb']['wiser_evmaidata_clicks'].find(filter=query,
        #                                                             limit=1,
        #                                                             projection=projection)
        result = client['wiseraidb']['wiser_evmaidata_clicks'].find_one(filter=query,
                                                                        limit=1,
                                                                        projection=projection)

        # Check if there is a result
        # return list(result)
        return result  # Return None if no result is found

    def make_recommendations(self, user_id, product_id, model_path):
        # Load the model and additional information
        try:
            checkpoint = torch.load(model_path)
        except:
            raise GenericAPIException(f'Train model file not available for this shop.',
                                      status_code=404)
        model = RecommenderNet(checkpoint['num_users'], checkpoint['num_items'],
                               checkpoint['embedding_size'], checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        user_to_index = checkpoint['user_to_index']
        item_to_index = checkpoint['item_to_index']
        min_date = checkpoint['min_date']
        max_date = checkpoint['max_date']

        # Map user and item to indices
        # print(user_to_index)
        user_index = user_to_index.get(user_id)
        if user_index is None:
            raise GenericAPIException(f'User not found in our train model.',
                                      status_code=404)

        item_index = item_to_index.get(product_id)
        if item_index is None:
            raise GenericAPIException(f'Product not found in our train model.',
                                      status_code=404)

        date = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
        scaled_date = (date - min_date) / (max_date - min_date)

        # Convert inputs to tensors
        user_index = torch.tensor([user_index], dtype=torch.long)
        item_index = torch.tensor([item_index], dtype=torch.long)
        scaled_date = torch.tensor([[scaled_date]], dtype=torch.float32)

        # Make a prediction using the loaded model
        with torch.no_grad():
            outputs = model(user_index, item_index, scaled_date)
            prediction = torch.argmax(outputs, dim=1)

        recommended_product = next((key for key, value in item_to_index.items() if value == int(prediction.item())),
                                   None)
        return {"product_id": recommended_product}


class DataWrangling():
    def read_specific_shop(self, shop_name):
        try:
            # Read the JSON file into a DataFrame
            data_df = pd.read_json(shop_name)
            return data_df
        except Exception as e:
            print(f"An error occurred while reading the JSON file: {e}")
            return None

    def train_test_split_df(self, df):
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        return train_data, test_data

    def scaling_date(self, train_df, test_df):
        # Get the earliest and latest click dates
        train_df['created_date'] = pd.to_datetime(train_df['cdate'])
        test_df['created_date'] = pd.to_datetime(test_df['cdate'])

        min_date = min(train_df['created_date'].min(), test_df['created_date'].min())
        max_date = max(train_df['created_date'].max(), test_df['created_date'].max())

        # Scale the click date between 0 and 1
        train_df['scaled_created_date'] = (train_df['created_date'] - min_date) / (max_date - min_date)
        test_df['scaled_created_date'] = (test_df['created_date'] - min_date) / (max_date - min_date)
        return train_df, test_df, min_date, max_date

    def user_item_matrix(self, df, train_df, test_df):

        # Create the user-item interaction matrix
        n_users = df['iplog'].nunique()
        n_items = df['product_id'].nunique()

        # Create a dictionary to map user IDs and product IDs to unique indices
        user_to_index = {user_id: i for i, user_id in enumerate(df['iplog'].unique())}
        item_to_index = {item_id: i for i, item_id in enumerate(df['product_id'].unique())}

        # Convert user and item IDs to indices
        train_df['user_index'] = train_df['iplog'].map(user_to_index)
        train_df['item_index'] = train_df['product_id'].map(item_to_index)
        test_df['user_index'] = test_df['iplog'].map(user_to_index)
        test_df['item_index'] = test_df['product_id'].map(item_to_index)
        return train_df, test_df


# Define the model architecture
class RecommenderNet(nn.Module):

    def __init__(self, num_users=None, num_items=None, embedding_size=None, num_classes=None):
        super(RecommenderNet, self).__init__()
        if num_users is not None and num_items is not None and embedding_size is not None and num_classes is not None:
            self.user_embedding = nn.Embedding(num_users, embedding_size)
            self.item_embedding = nn.Embedding(num_items, embedding_size)
            self.fc_layers = nn.Sequential(
                nn.Linear(embedding_size * 2 + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

    def forward(self, user_indices, item_indices, click_dates):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        concatenated = torch.cat([user_embedded, item_embedded, click_dates], dim=1)
        output = self.fc_layers(concatenated)
        return output
