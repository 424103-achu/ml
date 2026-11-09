import numpy as np
import pandas as pd

class CategoricalNB:
    def __init__(self, alpha=1.0): 
        self.alpha = alpha
        self.classes = []
        self.priors = {}
        self.likelihoods = {}
        self.categories = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for col in X.columns:
            self.categories[col] = np.unique(X[col])

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)

            self.likelihoods[c] = {}
            for col in X.columns:
                self.likelihoods[c][col] = {}
                V_col = len(self.categories[col])
                N_c_col = X_c[col].value_counts()
                total_count = len(X_c)

                for category in self.categories[col]:
                    numerator = N_c_col.get(category, 0) + self.alpha
                    denominator = total_count + (self.alpha * V_col)
                    self.likelihoods[c][col][category] = numerator / denominator
    
    def predict(self, X):
        predictions = []
        
        for _, x in X.iterrows():
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = 0
                for col in X.columns:
                    category = x[col] 
                    if category in self.likelihoods[c][col]:
                        likelihood += np.log(self.likelihoods[c][col][category])
                    else:
                        # Perfectly matches the denominator scaling used in fit
                        likelihood += np.log(self.alpha / (len(self.categories[col]) + self.alpha))
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions
