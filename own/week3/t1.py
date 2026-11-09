import numpy as np;
import pandas as pd;
def train_nb(X, y):
    classes = np.unique(y)
    n_samples, n_features = X.shape
    
    means = {}
    variances = {}
    priors = {}
    
    for c in classes:
        X_c = X[y == c]
    
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0) +1e-1
      
        priors[c] = X_c.shape[0] / n_samples 
        
    return {"classes": classes, "means": means, "variances": variances, "priors": priors}

def calculate_log_pdf(x, mean, variance):
    exponent = -((x - mean) ** 2) / (2 * variance)
    coefficient = -0.5 * np.log(2 * np.pi * variance)
    return coefficient + exponent

def predict_single(x, model):
    posteriors = {}
    
    for c in model["classes"]:
        prior_log = np.log(model["priors"][c])
      
        all_features_log_pdf = calculate_log_pdf(x, model["means"][c], model["variances"][c])
       
        conditional_log = np.sum(all_features_log_pdf)
      
        posteriors[c] = prior_log + conditional_log
   
    return max(posteriors, key=posteriors.get)

def predict_naive_bayes(X, model):
    return np.array([predict_single(x, model) for x in X])



df=pd.read_csv("SalaryData_Train.csv")
# print(df.head())
y_raw=df["Salary"]
# print(y_target)
x_raw=df.drop(columns=["Salary"])
print(x_raw)
x_encoded = pd.get_dummies(x_raw, dtype=float)  
x = x_encoded.to_numpy()
y = y_raw.to_numpy()
df2=pd.read_csv("SalaryData_Test.csv")
# print(df.head())
y_raw2=df["Salary"]
# print(y_target)
x_raw2=df.drop(columns=["Salary"])
print(x_raw2)
x_encoded2 = pd.get_dummies(x_raw2, dtype=float)  
x2 = x_encoded2.to_numpy()
y2 = y_raw2.to_numpy()
# print(x,y)
# 1. Train the model and get the dictionary
my_model = train_nb(x, y)

# 2. Predict using the same (or test) dataset
predictions = predict_naive_bayes(x2, my_model)

# 3. Print overall accuracy
accuracy = np.mean(predictions == y2)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
