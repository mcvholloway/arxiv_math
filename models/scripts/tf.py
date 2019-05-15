import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()

# Preprocesser
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

print("Reading in training set...")
articles = pd.read_csv('train.csv')

print("Preprocessing title and abstract...")
articles['title_and_abstract'] = (articles.title + ' ' + articles.abstract).progress_apply(preprocess_abstract)

tf_vect = TfidfVectorizer(stop_words = 'english')

print('Fitting TFIDFVectorizer')
tf_vect.fit(articles.title_and_abstract)

import pickle
print("Exporting model")
pickle.dump(tf_vect, open("tf_vect.pickle", "wb"))

