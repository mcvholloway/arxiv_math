def ValuePredictor(preprocessed,ft_predictor,mlb):
    #result = mlb.inverse_transform(pipeline.predict([preprocessed]))
    result = mlb.inverse_transform(ft_predictor.predict([preprocessed], min_one = True))
    print(result)
    if len(result[0]) > 0:
        result = [x for x in result[0]]
    # Make sure that at least one is predicted
    #else:
    #    result = [loaded_model.predict([preprocessed])[0][0][0].replace('__label__', '')]
    return result

def FindSimilar(to_predict_list, tf_vect, tfidf):
    import numpy as np
    from sklearn.metrics.pairwise import linear_kernel
    to_predict = ' '.join(to_predict_list).replace('\n', ' ')
    cosine_similarities = linear_kernel(tf_vect.transform([to_predict]), tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    related_docs_similarities = np.round(np.sort(cosine_similarities)[:-6:-1],2)
    return related_docs_indices, related_docs_similarities

def FindTags(to_predict_list, tf_vect, tfidf, search, df, yake):
    import numpy as np
    from sklearn.metrics.pairwise import linear_kernel
    to_predict = ' '.join(to_predict_list).replace('\n', ' ')
    indices = list(df.loc[(df.tags.apply(lambda x: search in x)) & (yake.yake_tags.apply(lambda x: search in x))].index)
    if len(indices) == 0: return list(), list()
    cosine_similarities = linear_kernel(tf_vect.transform([to_predict]), tfidf[indices,:]).flatten()
    related_docs_indices = list(np.array(indices)[cosine_similarities.argsort()[:-6:-1]])
    related_docs_similarities = np.round(np.sort(cosine_similarities)[:-6:-1],2)
    return related_docs_indices, related_docs_similarities


def make_lookup_dict():
    lookup_dict = {
                'AG' : 'math.AG - Algebraic Geometry',
                'AT' : 'math.AT - Algebraic Topology',
                'AP' : 'math.AP - Analysis of PDEs',
                'CT' : 'math.CT - Category Theory',
                'CA' : 'math.CA - Classical Analysis and ODEs',
                'CO' : 'math.CO - Combinatorics',
                'AC' : 'math.AC - Commutative Algebra',
                'CV' : 'math.CV - Complex Variables',
                'DG' : 'math.DG - Differential Geometry',
                'DS' : 'math.DS - Dynamical Systems',
                'FA' : 'math.FA - Functional Analysis',
                'GM' : 'math.GM - General Mathematics',
                'GN' : 'math.GN - General Topology',
                'GT' : 'math.GT - Geometric Topology',
                'GR' : 'math.GR - Group Theory',
                'HO' : 'math.HO - History and Overview',
                'IT' : 'math.IT - Information Theory',
                'KT' : 'math.KT - K-Theory and Homology',
                'LO' : 'math.LO - Logic',
                'MP' : 'math.MP - Mathematical Physics',
                'MG' : 'math.MG - Metric Geometry',
                'NT' : 'math.NT - Number Theory',
                'NA' : 'math.NA - Numerical Analysis',
                'OA' : 'math.OA - Operator Algebras',
                'OC' : 'math.OC - Optimization and Control',
                'PR' : 'math.PR - Probability',
                'QA' : 'math.QA - Quantum Algebra',
                'RT' : 'math.RT - Representation Theory',
                'RA' : 'math.RA - Rings and Algebras',
                'SP' : 'math.SP - Spectral Theory',
                'ST' : 'math.ST - Statistics Theory',
                'SG' : 'math.SG - Symplectic Geometry'}

    return lookup_dict

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
