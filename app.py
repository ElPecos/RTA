
import numpy as np
import pickle
from sklearn.datasets import load_iris
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
import pickle


iris = load_iris()

df = pd.DataFrame(data = np.c_[iris["data"], iris["target"]], columns = iris["feature_names"]+["target"])

df.drop(df.index[df["target"] == 2], inplace = True)
X = df.loc[:, ["petal length (cm)", "sepal length (cm)"]].values
y = df.loc[:, ["target"]].values


class Perceptron():
    def __init__(self, eta=0.01, n=20):
        self.eta=eta
        self.n=n
    def fit(self, X, y):
        self.w_=np.zeros(1+X.shape[1])
        self.errors_=[]
        for k in range(self.n):
            errors=0
            for xi, target in zip(X, y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    
model = Perceptron()
model.fit(X, y)

with open("perc_iris.pkl", "wb") as picklefile:
    pickle.dump(model, picklefile)
    
app = Flask(__name__)

@app.route('/api/v1.0/predict', methods = ["GET"])
def get_prediction():

    # sepal length
    sepal_length = float(request.args.get('sl'))
    # sepal width
    #sepal_width = float(request.args.get('sw'))
    # petal length
    petal_length = float(request.args.get('pl'))
    # petal width
    #petal_width = float(request.args.get('pw'))

    # The features of the observation to predict
    #features = [sepal_length,
    #            sepal_width,
    #            petal_length,
    #           petal_width]
    
    features = [sepal_length,
                petal_length]
    
    print(features)
    
    with open("perc_iris.pkl", "rb") as picklefile:
        model = pickle.load(picklefile)
    print(model)
    
    predicted_class = int(model.predict(features))
    
    return jsonify(features = features, predicted_class = predicted_class)


if __name__ == "__main__":
    app.run()
