import pandas as pd
import random
import torch
from torch_geometric.data import Data
import numpy as np


class GraphSampler:
    def __init__(self, dataset_path, veh_num, recommendation_size):
        self.recommended_movies = None
        self.dataset_path = dataset_path
        self.dataframe = pd.read_csv(dataset_path, sep=',', header=0,
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
        self.veh_num = veh_num
        # create user-movie dictionary
        self.user_movie_rating_dict = {}
        for index, row in self.dataframe.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            rating = row['rating']
            if user_id not in self.user_movie_rating_dict:
                self.user_movie_rating_dict[user_id] = []
            self.user_movie_rating_dict[user_id].append((movie_id, rating))
        # create a list to store the most rated movies among all users based on cache size
        self.recommendation_size = recommendation_size
        self.build_movie_cache()
        self.current_users = random.sample(list(self.user_movie_rating_dict.keys()), veh_num)
        self.node_features = None

    def get_recomended_movies(self):
        return [movies for movies, _ in self.recommended_movies]

    def get_node_features(self):
        return self.node_features


    def build_movie_cache(self):
        # Calculate average ratings for each movie
        movie_ratings = {}
        for user, movies in self.user_movie_rating_dict.items():
            for movie_id, rating in movies:
                if movie_id not in movie_ratings:
                    movie_ratings[movie_id] = []
                movie_ratings[movie_id].append(rating)

        # Create a list of movies sorted by average rating, limited by cache size
        self.recommended_movies = sorted(
            movie_ratings.items(),
            key=lambda item: sum(item[1]) / len(item[1]),  # average rating
            reverse=True
        )[:self.recommendation_size]

    def sample_movie(self):
        # Get all movies and ratings for this user
        node_features = [[self.recommended_movies[i][0] for i in range(self.recommendation_size)]]
        edge_weight = [[self.recommended_movies[i][1][0] for i in range(self.recommendation_size)]]
        edge_index = []
        for i in self.current_users:
            movies, ratings = zip(*self.user_movie_rating_dict[i])
            # Normalize ratings to probabilities
            total_rating = sum(ratings)
            probabilities = [r / total_rating for r in ratings]
            # Select one movie based on rating probabilities
            selected_movies = random.choices(list(zip(movies, ratings)), weights=probabilities,
                                             k=self.recommendation_size)
            movies = [movie for movie, _ in selected_movies]
            ratings = [rating for _, rating in selected_movies]
            node_features.append(movies)
            edge_weight.append(ratings)
        # Create edge index

        for i in range(1, self.veh_num + 1):
            edge_index.append((0, i))
            edge_index.append((i, 0))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.node_features = node_features
        node_features = torch.tensor(node_features, dtype=torch.float)  # Assuming movie IDs are integers
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        data = Data(x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_weight, dtype=torch.float)

        return data

    def update_current_user(self):
        # randomly get the index of the user to be replaced
        index = random.randint(0, self.veh_num - 1)
        # randomly select a new user
        new_user = random.choice(list(self.user_movie_rating_dict.keys()))
        self.current_users[index] = new_user

    def update_recommendation(self, recommended_movies):
        self.recommended_movies = recommended_movies


if __name__ == "__main__":
    dataset_path = '../ml-latest-small/ratings.csv'
    veh_num = 10
    sampler = GraphSampler(dataset_path, veh_num, 100)
    data = sampler.sample_movie()
    print(data)
