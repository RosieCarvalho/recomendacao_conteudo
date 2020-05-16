# coding: utf-8
import pandas as pd
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import argparse
import gc
import time
import numpy as np

# utils import
from fuzzywuzzy import fuzz


# https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
# outra opção: https://medium.com/@tomar.ankur287/item-item-collaborative-filtering-recommender-system-in-python-cf3c945fae1e

# codigo: https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/67f73f0b6bc20363a60d03beb88045d65538245d/movie_recommender/src/knn_recommender.py#L224

# ler os dados
def parse_args():
    parser = argparse.ArgumentParser(
        prog="Movie Recommender",
        description="Run KNN Movie Recommender")
    parser.add_argument('--path', nargs='?', default='recomendacao',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--movie_name', nargs='?', default='Toy Story (1995)',
                        help='provide your favoriate movie name')
    parser.add_argument('--top_n', type=int, default=20,
                        help='top n movie recommendations')
    parser.add_argument('--tags_filename', nargs='?', default='tags.csv')
    return parser.parse_args()


class KnnRecommender:
    def __init__(self, path_movies, path_ratings):
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = 0
        self.user_rating_thres = 0
        self.model = NearestNeighbors()

    def _prep_data(self):
        df_movies = pd.read_csv('movies.csv',
                                usecols=['movieId', 'title'],
                                dtype={'movieId': 'int32', 'title': 'str'})

        df_movies = pd.read_csv('movies.csv',
                                usecols=['movieId', 'title'],
                                dtype={'movieId': 'int32', 'title': 'str'})

        df_ratings = pd.read_csv('ratings.csv',
                                 usecols=['userId', 'movieId', 'rating'],
                                 dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        # filter data
        # conta o numero de avaliações do filme
        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])

        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))  # noqa
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # pivot and create movie-user matrix
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)

        # create mapper from movie title to index
        hashmap = {
            movie: i for i, movie in
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))  # noqa
            }

        # transform matrix to scipy sparse matrix
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
        # clean up
        del df_movies, df_movies_cnt, df_users_cnt
        del df_ratings, df_ratings_filtered, movie_user_mat
        gc.collect()

        return movie_user_mat_sparse, hashmap

    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        """
        defina o limite de frequência de classificação para filtrar filmes menos conhecidos e usuários menos ativos

        Parameters
        ----------
        movie_rating_thres: int, número mínimo de classificações recebidas pelos usuários
        user_rating_thres: int, número mínimo de classificações que um usuário fornece
        """
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):

        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _fuzzy_matching(self, hashmap, fav_movie):
        # print("\nHASHMAP")
        # print(hashmap)
        
        match_tuple = []
        # get match
        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Foram encontradas possíveis correspondências em nosso banco de dados: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap, fav_movie, n_recommendations):
        # fit
        # print("\nAQUI É DATA")
        # print(data)
        # print("\n")
        model.fit(data)
        # get input movie index
        print('You have input movie:', fav_movie)
        idx = self._fuzzy_matching(hashmap, fav_movie)
        print("\nIDX")
        print(idx)
        # inference
        print('Sistema de recomendação começa a fazer inferência')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        # print("\nINDICES")
        # print(indices)
        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]

        print('O meu sistema {: .2f} s fez inferência \n\
              '.format(time.time() - t0))
        # print('\nRAW')
        # print(raw_recommends)
        # return recommendation (movieId, distance)
        return raw_recommends

    def make_recommendations(self, fav_movie, n_recommendations):
        filmesRecomendados = []
        # get data
        movie_user_mat_sparse, hashmap = self._prep_data()

        # get recommendations
        raw_recommends = self._inference(
            self.model, movie_user_mat_sparse, hashmap,
            fav_movie, n_recommendations)
        # print results

        # print(hashmap)
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        # print('Recomendação for {}:'.format(fav_movie))
        for i, (idx, dist) in enumerate(raw_recommends):
            # print('{0}: {1}, with distance '
                #   'of {2}'.format(i+1, reverse_hashmap[idx], dist))
            filmesRecomendados.append(reverse_hashmap[idx])
        # print("\nREVERSE")
        # print(reverse_hashmap)
        return filmesRecomendados


def org(filmes_tags):
    ft = np.array(filmes_tags)
    z = np.zeros((len(np.unique(ft[:, 1])), len(np.unique(ft[:, 0]))))
    # print("AQUI\n", filmes_tags[0])  # np.unique(ft[:,0]))
    # print(ft[:, 1])
    df = pd.DataFrame(data=z, index=np.unique(
        ft[:, 1]), columns=np.unique(ft[:, 0]))
    print("\nORG")
    print(filmes_tags)
    # print(df.iloc[:5, :5])
    # for i in range(len(df.index)):
    #  for ii in range(len(df.columns)):
    #      print(ft[i], [df.columns[ii], df.index[i]])

    # print(i)
    # print(df.shape)


class KnnRecommenderTag:
    def __init__(self, path_movies, path_tags, movies_recommender):
        self.topN = movies_recommender
        self.path_movies = path_movies
        self.path_ratings = path_tags
        self.movies_recommender = movies_recommender
        self.model = NearestNeighbors()

    def _prep_data(self):
        #print('oi', self.movies_recommender)
        df_tags = pd.read_csv('tags.csv',
                              usecols=['movieId', 'tag'],
                              dtype={'movieId': 'int32', 'tag': 'str'}
                              )

        df_movies = pd.read_csv('movies.csv',
                                usecols=['movieId', 'title'],
                                dtype={'movieId': 'int32', 'title': 'str'})

        # aqui está como buscar os id dos filmes atraves dos nomes
        filmes_tags = []
        for movie in movies_recommender:
            idfilme = df_movies[df_movies['title'] ==
                                movie].loc[:, 'movieId'].to_numpy()
            # print('tags',df_tags[df_tags['movieId']==idfilme[0]])
            for ii in df_tags[df_tags['movieId'] == idfilme[0]].to_numpy():
                filmes_tags.append(ii)
            #filmes_tags = np.array(filmes_tags)
        # org(filmes_tags)
        # [:,1].shape)
        # aux1 = np.array(filmes_tags[0])
        # aux = pd.DataFrame(data=np.array([aux1[:,1],np.ones(len(filmes_tags[0]))]).T)#,columns=filmes_tags[:,0],index=filmes_tags[:,1])
        # aux2 = pd.Series(data=np.ones(len(filmes_tags[0])))#,columns=aux1[:,0]),index=aux1[:,1])
        # aux2 = column
        # print(aux2)
        # break
        # filmes_ids.append(df_movies[df_movies['title']==movie].loc[:,'movieId'].to_numpy())
        #dataFrame = pd.DataFrame(filmes_tags)

        #dataFrame.columns = ['movieId', 'tag']
        # movie_tag = dataFrame.pivot(
        #      index=0, columns=1, values=1, aggfunc='count').fillna(0)
        # print(dataFrame)


if __name__ == '__main__':
    print('agora vai!')
    # get args
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    tags_filename = args.tags_filename
    movie_name = args.movie_name
    # print("\nmovie_nAAAAAAAAAAame", movie_name)
    # print("\nMOVIE", movies_filename)
    top_n = args.top_n

    # initial recommender system
    recommender = KnnRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename))

    # set params
    recommender.set_filter_params(50, 50)
    recommender.set_model_params(20, 'brute', 'cosine', -1)

    # make recommendations
    movies_recommender = recommender.make_recommendations(movie_name, top_n)
    print('\nmovies_recommender')
    print(movies_recommender)

    print('\nmovies_recommender')
    # print(movies_recommender)
    
    recommenderTag = KnnRecommenderTag(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, tags_filename),
        movies_recommender
    )

    recommenderTag._prep_data()
