import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from MLRO import MLROSOversampler

tqdm.pandas()

print("Reading in data set...")
articles = pd.read_csv('train.csv')

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

articles['title_and_abstract'] = (articles.title + '. ' + articles.abstract).str.replace('\n', ' ')

import ast
def get_categories(categories):
    try:
      return ' '.join(['__label__' + x[5:] for x in ast.literal_eval(categories) if x[:5] == 'math.'])
    except:
      return 'Missing'

print("Creating labels...")
articles['labels'] = articles.categories.apply(get_categories)

import csv

print("Exporting...")
pd.DataFrame(articles.labels + ' ' + articles.title_and_abstract).dropna().to_csv('full_ft_train.csv', index = False, header = False,quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")

