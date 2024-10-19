
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
from sklearn.preprocessing import StandardScaler
from datacleaner import DataCleaner

cleaner = DataCleaner('./dataset/races.csv', './dataset/cyclists.csv')

class DataUnderstander:
    def __init__(self, dc):
        self.races_df = dc.races_df
        self.cyclist_df = dc.cyclist_df
        self.df = dc.df

    def correlate(self):
        correlations = {
            correlation_type: self.df.corr(numeric_only=True, method=correlation_type)
            for correlation_type in ("kendall", "pearson", "spearman")
        }

        for i, k in enumerate(correlations.keys()):
            correlations[k].loc[:, "correlation_type"] = k

        # Loop through each correlation type and create separate plots
        for corr_type, corr_matrix in correlations.items():
            corr_matrix = corr_matrix.drop(columns=["correlation_type"])

            # Create a new figure for each correlation type
            plt.figure(figsize=(12, 12))

            # Plot the heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

            # Set the title for the plot
            plt.title(f'{corr_type.capitalize()} Correlation')

            # Display the plot
            plt.show()
    
    def normalize(self):
        numeric_columns = ['points', 'length', 'climb_total', 'profile', 'startlist_quality', 'cyclist_age', 'delta', 'birth_year', 'weight', 'height']
        scaler = StandardScaler() #Si prova con la zscore ora eh 
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])

    

dm = DataUnderstander(cleaner)
dm.normalize()
print(dm.correlate())