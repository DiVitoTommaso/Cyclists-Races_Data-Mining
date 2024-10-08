
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
from sklearn.preprocessing import StandardScaler


class DataMiner:
    def __init__(self, dataset_csv):
        self.df = pd.read_csv(dataset_csv, parse_dates=["date"])

    def correlate(self):
        correlations = {
              correlation_type: self.df.corr(numeric_only=True, method=correlation_type)
               for correlation_type in ("kendall", "pearson", "spearman")
        }
        for i, k in enumerate(correlations.keys()):
           correlations[k].loc[:, "correlation_type"] = k
        correlations_matrix = pd.concat(correlations.values())
        return correlations_matrix
    
    def normalize(self):
        numeric_columns = ['points', 'length', 'climb_total', 'profile', 'startlist_quality', 'cyclist_age', 'delta', 'birth_year', 'weight', 'height']
        scaler = StandardScaler() #Si prova con la zscore ora eh 
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        #print(self.df.head())

    

dm = DataMiner("./dataset/dataset.csv")
dm.normalize()
print(dm.correlate())