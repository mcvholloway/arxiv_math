class MLROSOversampler:
    def __init__(self):
        pass
        
    def IRLbl(self, dataset, label, labelset):
        ''' calculates the imbalance ratio per label
        needs dataset with dummy columns '''
        num = max(dataset.loc[:,labelset].sum())
        denom = dataset.loc[:,label].sum()
        return num/denom
    
    def MeanIR(self,dataset, labelset):
        ''' calculates the mean imbalance ratio'''
        import numpy as np
        return np.mean([self.IRLbl(dataset, label, labelset) for label in labelset])
    
    def MLROS(self, dataset, labels, percentage, batch_size = 100):
        from numpy import random
        import pandas as pd
        starting_size = len(dataset)
        samplesToClone = int(len(dataset) * percentage / 100)
        mir = self.MeanIR(dataset, labels)
        cloners = [label for label in labels if self.IRLbl(dataset, label, labels) > mir]
        clone_sets = [dataset.loc[dataset[label] == 1].reset_index(drop = True) for label in cloners]
        clone_set_lengths = [len(x) for x in clone_sets]


        cloneCount = 0

        while(cloneCount < samplesToClone and len(cloners) > 0):
            clones = pd.DataFrame()
            for i,label in enumerate(cloners):
                clones = clones.append(clone_sets[i].loc[random.choice(range(clone_set_lengths[i]), batch_size, replace = True)])
                #clones = clones.append(clone_sets[i].loc[random.choice(range(len(clone_sets[i])))])
            cloneCount += batch_size * len(cloners)
            #print(len(clones))
            print(cloneCount, '/', samplesToClone )

            dataset = dataset.append(clones, ignore_index = True)

            for label in cloners:
                if self.IRLbl(dataset, label, labels) <= mir:
                    idx = cloners.index(label)
                    cloners = cloners[:idx] + cloners[idx+1:]
                    clone_sets = clone_sets[:idx] + clone_sets[idx+1:]
                    clone_set_lengths = clone_set_lengths[:idx] + clone_set_lengths[idx+1:]

        return dataset
