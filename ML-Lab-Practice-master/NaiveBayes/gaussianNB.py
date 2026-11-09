import numpy as np
class GaussianNB:
  def __init__(self):
    self.classes = []
    self.priors = {}
    self.means = {}
    self.variances = {}
    
  def fit(self, X, y):
    self.classes = np.unique(y)

    for c in self.classes:
      X_c = X[y == c]
      self.priors[c] = len(X_c) / len(X)
      self.means[c] = np.mean(X_c, axis=0)
      self.variances[c] = np.var(X_c, axis=0) + 1e-9

  def gaussian_pdf(self, x, mean, variance):
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
    denominator = np.sqrt(2 * np.pi * variance)
    return exponent / denominator

  def predict(self, X):
    predictions = []
    for x in X.values:
      posteriors = []

      for c in self.classes:
        prior = np.log(self.priors[c])
        posterior = np.sum(np.log(self.gaussian_pdf(x, self.means[c], self.variances[c])+1e-9))
        posterior = prior + posterior
        posteriors.append(posterior)

      predictions.append(self.classes[np.argmax(posteriors)])

    return predictions

    