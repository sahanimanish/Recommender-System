import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def getDataFrame():
    df = pd.read_csv('./data/ml-latest-small/ratings.csv')
    df_links = pd.read_csv('./data/ml-latest-small/links.csv')
    df_titles = pd.read_csv('./data/ml-latest-small/movies.csv')
    return df, df_links, df_titles
class Recommender:
    def __init__(self, df, df_links, df_titles):
        self.df = df
        self.df_titles = df_titles
        self.df_links = df_links
        self.n_users = df.userId.unique().shape[0]
        self.n_items = df.movieId.unique().shape[0]
        self.ratings = np.zeros((self.n_users, self.n_items))
        # now we are just sorting the values of moviesid so that we can generate 
        # #index corresponding to them
        self.movie_Id= np.sort(df.movieId.unique()) # movie id in sorted way
        self.movie_index_Id = dict(map(lambda t: (t[1], t[0]), enumerate(self.movie_Id)))
        self.data = np.array(self.df, dtype='int')

        for row in self.data:
            self.ratings[row[0]-1, self.movie_index_Id[row[1]]] = row[2]
        
        self.arr_links = np.array(self.df_links, dtype='int')

        # maping movieId to imdbId
        self.movieId_to_imdbId={}
        for row in self.arr_links:
            self.movieId_to_imdbId[row[0]] = "tt0"+str(row[1])

        self.train, self.test = self.train_test_split()
        self.item_similarity=self.cosine_sim( kind='item')  # for item-item sim
        # data dictionary of movie titles
        self.movie_title_to_id = {}
        self.movie_df_arr = np.array(self.df_titles)
        for i in range(len(self.movie_df_arr)):
            self.movie_title_to_id[self.movie_df_arr[i][1]] = self.movie_df_arr[i][0]

    def train_test_split(self):
        test = np.zeros(self.ratings.shape)
        train = self.ratings.copy()
        for user in range(self.ratings.shape[0]):
            test_ratings = np.random.choice(self.ratings[user, :].nonzero()[0], size=10, replace=False)
        # here replace=False means no value should be repeated in choice all should be different
        train[user, test_ratings]=0  # these random selected movies rating make 0
        test[user, test_ratings] = self.ratings[user, test_ratings]
        assert(np.all(train*test)==0)
        return train, test
    # cosine similarity
    # epsilon - small number for handling divide by zero error
    def cosine_sim(self, kind='user', epsilon=1e-9):
        if kind=='user':
            sim= self.ratings.dot(self.ratings.T)+epsilon
        if kind=='item':
            sim= (self.ratings.T).dot(self.ratings)+epsilon
            norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim/norms/norms.T)
    # user_similarity=cosine_sim(ratings)  # for user-user sim  
    def get_mse(self, pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)
    
    def predict_topk(self, kind='user', k=40):
        pred = np.zeros(self.ratings.shape)
        # train, test = train_test_split(self.ratings)
        # item_similarity=cosine_sim(self.ratings, kind='item')  # for item-item sim
        if kind == 'user':
            for i in range(self.ratings.shape[0]):
                top_k_users = [np.argsort(self.item_similarity[:,i])[:-k-1:-1]]
                for j in range(self.ratings.shape[1]):
                    pred[i, j] = self.item_similarity[i, :][top_k_users].dot(self.ratings[:, j][top_k_users]) 
                    pred[i, j] /= np.sum(np.abs(self.item_similarity[i, :][top_k_users]))
        if kind == 'item':
            for j in range(self.ratings.shape[1]):
                top_k_items = [np.argsort(self.item_similarity[:,j])[:-k-1:-1]]
                for i in range(self.ratings.shape[0]):
                    pred[i, j] = self.item_similarity[j, :][top_k_items].dot(self.ratings[i, :][top_k_items].T) 
                    pred[i, j] /= np.sum(np.abs(self.item_similarity[j, :][top_k_items]))        
        return pred
    def top_k_movies(self, movie_idx, k=6):
        return [self.movieId_to_imdbId[self.movie_Id[x]] for x in np.argsort(self.item_similarity[movie_idx,:])[:-k-1:-1]]
    def top_k_movies_by_title(self, title, k=6):
        movieid = self.movie_title_to_id[title]
        movie_idx = self.movie_index_Id[movieid]
        return [self.movieId_to_imdbId[self.movie_Id[x]] for x in np.argsort(self.item_similarity[movie_idx,:])[:-k-1:-1]]

    def display(self):
        print(self.movie_title_to_id["Up Close and Personal (1996)"])















# pred_k_item = predict_topk(train, item_similarity, kind='item', k=40)


# df, df_links, df_titles =  getDataFrame()
# rem = Recommender(df, df_links, df_titles)
# rem.display()
