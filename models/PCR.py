import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class PCRModel:
    def __init__(self, n_components=10):
        """
        Initialize the PCRModel with the specified number of principal components.
        :param n_components: Number of principal components to use in PCA
        """
        self.n_components = n_components
        self.pcr = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('linear_regression', LinearRegression())
        ])

    def fit(self, X_train, y_train):
        """
        Fit the PCR model to the training data.
        :param X_train: Training feature data
        :param y_train: Training target data
        """
        self.pcr.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using the PCR model on the test data.
        :param X_test: Test feature data
        :return: Predicted values
        """
        return self.pcr.predict(X_test)

    def calculate_accuracy(self, y_test, y_pred):
        """
        Calculate the accuracy of the model.
        :param y_test: True target values
        :param y_pred: Predicted target values
        :return: Accuracy score
        """
        y_pred_int = np.round(y_pred).astype(int)
        y_test_int = y_test.astype(int)
        accuracy = accuracy_score(y_test_int, y_pred_int)
        return accuracy

# Example Usage
# pcr_model = PCRModel(n_components=10)
# pcr_model.fit(X_train, y_train)
# y_pred = pcr_model.predict(X_test)
# accuracy = pcr_model.calculate_accuracy(y_test, y_pred)
# print(f"PCR accuracy: {np.round(accuracy*100, 2)}%")
