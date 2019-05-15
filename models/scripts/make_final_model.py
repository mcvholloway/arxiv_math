import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from MLRO import MLROSOversampler

tqdm.pandas()

print("Reading in training set...")
articles = pd.read_csv('../data/arxiv_math.csv')

def preprocess_abstract(abstract):
    import re
    abstract = abstract.replace('\n', ' ') #remove new line characters
    abstract = abstract.replace('$K$-theory', 'k-theory').replace('$C^*$-algebra', 'C-algebra').replace('\\emph', '') #fix a few common
    abstract = re.sub('\$.*?\$', '', abstract) #remove math
    bstract = re.sub('\[.*?\]', '', abstract) #remove anything in brackets
    abstract = re.sub('\s[a-zA-Z]{1}\s', ' ', abstract) #remove single letters - eg. consider a group G - the G does not add anything
    abstract = re.sub('\s[0-9]+\s', ' ', abstract) #remove any single numbers
    abstract = re.sub('\(.*?\)', '', abstract) #remove parentheses
    abstract = re.sub('\s[A-Z]{1}\.\s', ' ', abstract) #remove first initials
    abstract = abstract.replace('*', '').replace('{', '').replace('}', '')
    abstract = re.sub(' +', ' ', abstract) #remove extra spaces
    return abstract

print('Preprocessing...')
articles['title_and_abstract'] = (articles.title + ' ' + articles.abstract).progress_apply(preprocess_abstract)

articles = articles.drop(columns = ['title', 'abstract'])

def get_math_categories(categories):
    import ast
    return [x[5:] for x in ast.literal_eval(categories) if x[:5] == 'math.']

print("Finding Categories...")
articles['math_categories'] = articles.categories.progress_apply(get_math_categories)

mlb = MultiLabelBinarizer()
mlb.fit(articles['math_categories'])

articles = pd.concat([articles, pd.DataFrame(mlb.transform(articles.math_categories), columns = mlb.classes_)], axis =1)

mso_resampler = MLROSOversampler()

print('Resampling...')
articles = mso_resampler.MLROS(articles, labels = mlb.classes_, percentage = 25)

X = articles.title_and_abstract
y = articles[mlb.classes_]

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline( steps =
                    [
                        ('vectorizer', TfidfVectorizer(stop_words = 'english')),
                     ('clf', OneVsRestClassifier(LogisticRegression()))
                    ])

print('Fitting Model...')
pipeline.fit(X, y)

print('Pickling Model...')
import pickle
pkl_filename = "model_pipeline.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(pipeline, file)

pkl_filename = "mlb.pkl"                
with open(pkl_filename, 'wb') as file:
    pickle.dump(mlb, file)


