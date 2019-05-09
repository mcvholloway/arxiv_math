def ValuePredictor(to_predict_list, loaded_model):
    to_predict = '. '.join(to_predict_list).replace('\n', ' ')
    result = loaded_model.predict([to_predict], threshold = 0.5, k = 6)
    # Make sure that at least one is predicted
    if len(result[0][0]) == 0:
        result = loaded_model.predict([to_predict])
    return result[0][0]

def FindSimilar(to_predict_list, tf_vect, tfidf):
    import numpy as np
    from sklearn.metrics.pairwise import linear_kernel
    to_predict = ' '.join(to_predict_list).replace('\n', ' ')
    cosine_similarities = linear_kernel(tf_vect.transform([to_predict]), tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    related_docs_similarities = np.round(np.sort(cosine_similarities)[:-6:-1],2)
    return related_docs_indices, related_docs_similarities

def FindTags(to_predict_list, tf_vect, tfidf, search, df):
    import numpy as np
    from sklearn.metrics.pairwise import linear_kernel
    to_predict = ' '.join(to_predict_list).replace('\n', ' ')
    indices = list(df.loc[df.tags.apply(lambda x: search in x)].index)
    if len(indices) == 0: return list(), list()
    cosine_similarities = linear_kernel(tf_vect.transform([to_predict]), tfidf[indices,:]).flatten()
    related_docs_indices = list(np.array(indices)[cosine_similarities.argsort()[:-6:-1]])
    related_docs_similarities = np.round(np.sort(cosine_similarities)[:-6:-1],2)
    return related_docs_indices, related_docs_similarities


def make_lookup_dict():
    lookup_dict = {
                '__label__AG' : 'math.AG - Algebraic Geometry',
                '__label__AT' : 'math.AT - Algebraic Topology',
                '__label__AP' : 'math.AP - Analysis of PDEs',
                '__label__CT' : 'math.CT - Category Theory',
                '__label__CA' : 'math.CA - Classical Analysis and ODEs',
                '__label__CO' : 'math.CO - Combinatorics',
                '__label__AC' : 'math.CA - Commutative Algebra',
                '__label__CV' : 'math.CV - Complex Variables',
                '__label__DG' : 'math.DG - Differential Geometry',
                '__label__DS' : 'math.DS - Dynamical Systems',
                '__label__FA' : 'math.FA - Functional Analysis',
                '__label__GM' : 'math.GM - General Mathematics',
                '__label__GN' : 'math.GN - General Topology',
                '__label__GT' : 'math.GT - Geometric Topology',
                '__label__GR' : 'math.GR - Group Theory',
                '__label__HO' : 'math.HO - History and Overview',
                '__label__IT' : 'math.IT - Information Theory',
                '__label__KT' : 'math.KT - K-Theory and Homology',
                '__label__LO' : 'math.LO - Logic',
                '__label__MP' : 'math.MP - Mathematical Physics',
                '__label__MG' : 'math.MG - Metric Geometry',
                '__label__NT' : 'math.NT - Number Theory',
                '__label__NA' : 'math.NA - Numerical Analysis',
                '__label__OA' : 'math.OA - Operator Algebras',
                '__label__OC' : 'math.OC - Optimization and Control',
                '__label__PR' : 'math.PR - Probability',
                '__label__QA' : 'math.QA - Quantum Algebra',
                '__label__RT' : 'math.RT - Representation Theory',
                '__label__RA' : 'math.RA - Rings and Algebras',
                '__label__SP' : 'math.SP - Spectral Theory',
                '__label__ST' : 'math.ST - Statistics Theory',
                '__label__SG' : 'math.SG - Symplectic Geometry'}

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
