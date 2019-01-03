
# coding: utf-8

# # Cluster Analysis
# > Use t-SNE analysis to explore similar movies

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


# > Load data using `pandas.read_csv`

# In[ ]:


df = pd.read_csv('movies_metadata.csv', encoding='latin1')
df['released'] = pd.to_datetime(df.release_date)
df.head()


# In[ ]:


name = df.pop('original_title')


# In[ ]:


class OHE(BaseEstimator, TransformerMixin):
    """Perform one-hot encoding of categorical data"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        self.cat = X[self.col].astype('category').cat.categories
        return self
    
    def transform(self, X, y=None):
        return pd.get_dummies(X[self.col].astype('category', categories=self.cat))


# In[ ]:


class Take(BaseEstimator, TransformerMixin):
    """Pass through a single column without modification"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.col].to_frame(self.col)


# In[ ]:


class TimeToEpoch(BaseEstimator, TransformerMixin):
    """Convert a datetime column into seconds-since-epoch"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.col].astype(int).to_frame(self.col)


# In[ ]:
#Index(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
#       'imdb_id', 'original_language', 'original_title', 'overview',
#       'popularity', 'poster_path', 'production_companies',
#       'production_countries', 'release_date', 'revenue', 'runtime',
#       'spoken_languages', 'status', 'tagline', 'title', 'video',
#       'vote_average', 'vote_count', 'year', 'description', 'cast', 'crew',
#       'keywords', 'cast_size', 'crew_size', 'director', 'soup'],
#      dtype='object')

#adult, genres,original_language,
features = [
    ('budget', Take('budget')),
    ('gross', Take('gross')),
    ('votes', Take('votes')),
    ('cast_size', Take('cast_size')),
    ('crew_size', Take('crew_size')),
    ('year', Take('year')), 
    ('revenue', Take('revenue')),
    ('runtime', Take('runtime')),
    ('vote_average', Take('vote_average')),
    ('vote_count', Take('vote_count')),
    ('popularity', Take('popularity')),
   
    ('genres', OHE('genres')),
    ('adult', OHE('adult')),
    ('original_language', OHE('original_language')),
    ('spoken_languages', OHE('spoken_languages')),
    ('cast', OHE('cast')),
    ('crew', OHE('crew')),
    ('director', OHE('director')),
    ('production_countries', OHE('production_countries')),
    ('production_companies', OHE('production_companies')),
    ('time', TimeToEpoch('released'))
    #('soup', OHE('soup')),
    #('keywords', OHE('keywords'))
    ]
pipe = Pipeline([
    ('feat', FeatureUnion(features)),
    ('scale', StandardScaler())
])

trans = pipe.fit_transform(df)


# In[ ]:


# Try to cluster using KMeans for colouring out plot
cluster = KMeans(n_clusters=20)
group_pred = cluster.fit_predict(trans)

# Perform t-SNE to reduce the dimensionality down to 2 dimenions, for easier plotting.
tsne = TSNE(n_components=2)
tsne_fit = tsne.fit_transform(trans)


# In[ ]:


init_notebook_mode(connected=True)

trace = go.Scatter(
    x=tsne_fit.T[0], 
    y=tsne_fit.T[1],
    mode='markers',
    name='Lines, Markers and Text',
    text=name,
    textposition='top',
    marker=dict(
        color = group_pred, #set color equal to a variable
        colorscale='Portland',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

