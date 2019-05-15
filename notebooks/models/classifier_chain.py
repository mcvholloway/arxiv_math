from scipy import sparse
import numpy as np

class ClassifierChain:
    def __init__(self, model):
        #pass
        self.model = model

    def fit(self, X, y):
        counts = y.sum(axis = 0)
        self.order_ = np.argsort(counts)[::-1]
        models = [None] * y.shape[1]
        
        for i in self.order_:
            #model = LinearSVC()
            model = self.model()
            models[i] = model.fit(X, y[:,i])
            try: 
                X = np.hstack([X, model.predict(X).reshape(-1,1)])
            except:
                X = sparse.hstack([X, model.predict(X).reshape(-1,1)])
        
        self.models_ = models
        
    def predict(self, X):
        predictions = [None]*len(self.order_)
        for i in self.order_:
            prediction = (self.models_[i].predict(X)).reshape(-1,1)
            predictions[i] = prediction
            try:
                X = np.hstack([X, prediction])
            except:
                X = sparse.hstack([X, prediction])
            
        return np.concatenate(predictions, axis = 1)  
