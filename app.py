import json
import requests
import recommenderModel as rm
from flask import Flask, render_template, url_for
import pickle
df, df_links, df_titles = rm.getDataFrame()
# recommender = rm.Recommender(df, df_links, df_titles)

# duming data
# fl = open('recommenderdb.pkl', 'wb')
# pickle.dump(recommender, fl)
# fl.close()
# end

# # loading pickle
fl = open('recommenderdb.pkl', 'rb')
rmObj = pickle.load(fl)
fl.close()


# movies = rmObj.top_k_movies("Up Close and Personal (1996)")
# print(movies)
# a = [1, 2, 3]

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/<int:id>')
# def getMovies(id):
#     movies = rmObj.top_k_movies(id)
#     print(movies)
#     links = []
#     for movie in movies:
#         movie = movie.strip()
#         url = f"http://www.omdbapi.com/?i={movie}&apikey=df614cab"
#         # url = f"http://www.omdbapi.com/?i={movie}&apikey=326b93af"
#         # url = f"http://img.omdbapi.com/?apikey=326b93af&i={movie}"
#         res = requests.get(url)
#         res = json.loads(res.content.decode('utf-8'))
#         links.append(res)
#     print(links)
#     return render_template('movies.html', name=links)

@app.route('/<string:title>')
def getMovies(title):
    # movies = rmObj.top_k_movies(id)
    movies = rmObj.top_k_movies_by_title(title)
    # print(movies)
    links = []
    for movie in movies:
        url = f"http://www.omdbapi.com/?i={movie}&apikey=df614cab"
        # url = f"http://www.omdbapi.com/?i={movie}&apikey=57024814"
        # url = f"http://www.omdbapi.com/?i={movie}&apikey=326b93af"
        # url = f"http://img.omdbapi.com/?apikey=326b93af&i={movie}"
        print(movie)
        res = requests.get(url)
        res = json.loads(res.content.decode('utf-8'))
        links.append(res)
    print(links)
    return render_template('movies.html', name=links)

if __name__ == "__main__":
    app.run(debug=True)
