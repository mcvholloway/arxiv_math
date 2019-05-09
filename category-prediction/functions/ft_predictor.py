class FTPredictor():
    def __init__(self, ft_model):
        self.model = ft_model
        
    def fit(self, X, y = None):
        return self
    
    def predict(self, X, y = None, min_one = False):
        import numpy as np
        predictions = []
        for x in X: 
            try:
                prediction = [label[-2:] for label in self.model.predict(x, threshold = 0.5, k = 6)[0]]
            except:
                prediction = list()
            if len(prediction) == 0 and min_one:
                prediction = [label[-2:] for label in self.model.predict(x)[0]]
            prediction = self.predict_class(prediction)
            predictions.append(prediction)
        return np.array(predictions)
    
    def predict_class(self, prediction):
        predicted_classes = []
        for category in ['AC', 'AG', 'AP', 'AT', 'CA', 'CO', 'CT', 'CV', 'DG', 'DS', 'FA',
       'GM', 'GN', 'GR', 'GT', 'HO', 'IT', 'KT', 'LO', 'MG', 'MP', 'NA',
       'NT', 'OA', 'OC', 'PR', 'QA', 'RA', 'RT', 'SG', 'SP', 'ST']:
            if category in prediction:
                predicted_classes.append(1)
            else:
                predicted_classes.append(0)
        return predicted_classes
    
    def predict_proba(self, X, y = None):
        import numpy as np
        predictions = []
        for x in X:
            probas = list(self.model.predict([x], threshold=-1, k = 32)[1][0])
            labels = list(self.model.predict([x], threshold=-1, k = 32)[0][0])
            ordered_probas = []
            for category in ['AC', 'AG', 'AP', 'AT', 'CA', 'CO', 'CT', 'CV', 'DG', 'DS', 'FA',
           'GM', 'GN', 'GR', 'GT', 'HO', 'IT', 'KT', 'LO', 'MG', 'MP', 'NA',
           'NT', 'OA', 'OC', 'PR', 'QA', 'RA', 'RT', 'SG', 'SP', 'ST']:
                ordered_probas.append(probas[labels.index('__label__' + category)])
            predictions.append(ordered_probas)
        return np.array(predictions)
