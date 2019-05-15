import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from MLRO import MLROSOversampler

tqdm.pandas()

print("Reading in data set...")
articles = pd.read_csv('arxiv_math.csv')

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

import csv

print("Exporting...")
pd.DataFrame(articles.title_and_abstract).dropna().to_csv('ft_vectors.csv', index = False, header = False,quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")

