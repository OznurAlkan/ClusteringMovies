import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.cluster import KMeans

#Index(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
#       'imdb_id', 'original_language', 'original_title', 'overview',
#       'popularity', 'poster_path', 'production_companies',
#       'production_countries', 'release_date', 'revenue', 'runtime',
#       'spoken_languages', 'status', 'tagline', 'title', 'video',
#       'vote_average', 'vote_count'],
#      dtype='object')

#Index(['budget', 
#       'popularity', 'production_companies',
#       'production_countries', 'revenue', 'runtime',
#       'vote_average', 'vote_count'],
#      dtype='object')

df = pd.read_csv('../data/movies_metadata.csv', encoding='latin1')
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
df = df[['id','genres','original_language', 'spoken_languages', 'production_companies','production_countries','budget', 'popularity','revenue', 'runtime','vote_average', 'vote_count','year']]
df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['genres'] = df['genres'].apply(lambda x: ','.join(x))
x = df.set_index('id').genres.str.split(r',', expand=True).stack().reset_index(level=1, drop=True).to_frame('genres'); 
genre_data = pd.get_dummies(x, prefix='g', columns=['genres']).groupby(level=0).sum()
df = df.set_index('id')

df_genre = pd.merge(genre_data, df, left_index=True, right_index=True)
df_genre = df_genre.drop(['genres'], axis=1)

original_language_data = pd.get_dummies(df_genre, columns=["original_language"], prefix=["ol"])
spoken_languages_data = pd.get_dummies(original_language_data, columns=["spoken_languages"], prefix=["sl"])
production_companies_data = pd.get_dummies(spoken_languages_data, columns=["production_companies"], prefix=["pcom"])
production_countries_data = pd.get_dummies(production_companies_data, columns=["production_countries"], prefix=["pcou"])

temp2 = production_countries_data.head(100)
kmeans = KMeans(n_clusters=20)
group_pred = kmeans.fit_predict(temp2)

mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

dictlist = []
for key, value in mydict.items():
    temp = [key,value]
    dictlist.append(temp)
