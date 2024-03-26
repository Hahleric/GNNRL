"""
take in the raw data and preprocess it into the format that can be used by the model
since this data will be used to train the recommender model, so we need a pseudo recommend list
here as well
"""

import numpy as np
import pandas as pd


def get_ratings(dataset_path):
    """
    read the file and return the data
    :param dataset_path: file path
    :return: ratings
    """
    ratings_file = dataset_path + '/ratings.dat'
    ratings = pd.read_csv(ratings_file, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                          engine='python')
    ratings = ratings.drop(['Timestamp'], axis=1)
    return ratings


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
        ratings_file_path,
        sep='::',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        engine='python'
    )
    # 创建数据透视表
    rating_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')

    # 替换NaN为0
    rating_matrix.fillna(0, inplace=True)
    # 转换为ndarray
    rating_ndarray = rating_matrix.to_numpy()

    return rating_ndarray


def create_top_100_rating_matrix(ratings_file_path):
    """
    Creates a rating matrix from the MovieLens ratings file,
    including only the top 100 rated movies for each user.

    Parameters:
    ratings_file_path (str): The file path to the MovieLens ratings.dat file.

    Returns:
    np.ndarray: A 2D numpy array where rows represent users, columns represent the top 100 movies for each user,
                and values represent ratings. Only the top 100 movies rated by each user are included.
    """
    # 读取评分数据
    ratings = pd.read_csv(
        ratings_file_path,
        sep='::',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        engine='python'
    )

    # 创建数据透视表
    rating_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')

    # 将NaN替换为0
    rating_matrix.fillna(0, inplace=True)

    # 为每个用户保留评分最高的前100部电影
    top_100_rating_matrix = rating_matrix.apply(lambda x: x.nlargest(100), axis=1).fillna(0)

    # 转换为ndarray
    top_100_rating_ndarray = top_100_rating_matrix.to_numpy()

    return top_100_rating_ndarray

def get_user_rated_movies(ratings_file_path):
    """
    Returns a dictionary where each key is a user ID and the value is a list of movie IDs
    that the user has rated.

    Parameters:
    ratings_file_path (str): The file path to the MovieLens ratings.dat file.

    Returns:
    dict: A dictionary with user IDs as keys and lists of rated movie IDs as values.
    """
    # 读取评分数据
    ratings = pd.read_csv(
        ratings_file_path,
        sep='::',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        engine='python'
    )

    # 分组按用户ID，获取每个用户评分的电影列表
    user_rated_movies = ratings.groupby('UserID')['MovieID'].apply(list).to_dict()

    return user_rated_movies


if __name__ == "__main__":
    user_movie = get_user_rated_movies('../ml-10M100K/ratings.dat')
    print(len(user_movie))
