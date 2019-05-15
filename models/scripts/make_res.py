import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from MLRO import MLROSOversampler

tqdm.pandas()

print("Reading in training set...")
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

print('Exporting...')
articles.to_csv('train_res.csv', index = False)

