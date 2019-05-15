import pandas as pd

print("Importing train set...")
articles = pd.read_csv('train_res.csv')

print("Merging title and abstract...")
articles['title_and_abstract'] = (articles.title + ' ' + articles.abstract)

articles.title_and_abstract = articles.title_and_abstract.str.replace('\n', ' ')

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
pd.DataFrame(articles.labels + ' ' + articles.title_and_abstract).dropna().to_csv('ft_train_res.csv', index = False, header = False,quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
