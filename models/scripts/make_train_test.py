import pandas as pd
from sklearn.model_selection import train_test_split

print('Reading in dataset...')
articles = pd.read_csv('arxiv_math.csv')

train, test = train_test_split(articles)

print('Exporting...')
train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)
