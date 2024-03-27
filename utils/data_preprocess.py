"""
take in the raw data and preprocess it into the format that can be used by the model
since this data will be used to train the recommender model, so we need a pseudo recommend list
here as well
"""

import numpy as np
import pandas as pd


def create_rating_matrix(ratings_file_path):
    """
    Creates a rating matrix from the MovieLens ratings file.

    Parameters:
    ratings_file_path (str): The file path to the MovieLens ratings.dat file.

    Returns:
    np.ndarray: A 2D numpy array where rows represent users, columns represent movies,
                and values represent ratings. Unrated movies are represented by 0.
    """
    # 读取评分数据
    ratings = pd.read_csv(
        ratings_file_path
    )
    # 创建数据透视表
    rating_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

    # 替换NaN为0
    rating_matrix.fillna(0, inplace=True)
    # 转换为ndarray
    rating_ndarray = rating_matrix.to_numpy()
    sorted_ratings = ratings.sort_values(by=['userId', 'rating'], ascending=[True, False])
    top_movies_by_user = sorted_ratings.groupby('userId').head(100)
    top_movies_by_user['rank'] = top_movies_by_user.groupby('userId')['rating'].rank(ascending=False, method='first')
    user_movie_matrix = top_movies_by_user.pivot(index='userId', columns='rank', values='movieId')
    user_rated_movies = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    # Count the number of ratings for each movie
    movie_counts = ratings['movieId'].value_counts()

    # Get the top 100 most rated movies
    top_100_popular_movies = movie_counts.nlargest(100).index.tolist()

    return rating_ndarray, user_movie_matrix.to_numpy(), user_rated_movies, top_100_popular_movies


if __name__ == "__main__":
    rating_ndarray, top_100_rating_ndarray, user_rated_movies, top_100_popular_movies = create_rating_matrix('../ml-latest-small/ratings.csv')
