import pandas as pd
from tqdm import tqdm

tqdm.pandas()

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


print("Importing train set...")
articles = pd.read_csv('train.csv')

print("Preprocessing title and abstract...")
articles['title_and_abstract'] = (articles.title + ' ' + articles.abstract).progress_apply(preprocess_abstract)

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
pd.DataFrame(articles.labels + ' ' + articles.title_and_abstract).dropna().to_csv('ft_train.csv', index = False, header = False,quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
