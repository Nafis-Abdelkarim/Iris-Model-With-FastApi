# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

# Declearing FASTAPI Instance 
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

#Loading Iris Dataset
iris = load_iris()

#Getting our features and Classes
X = iris.data
Y = iris.target

#Create and Fitting our Model
clf = GaussianNB()
clf.fit(X, Y)

# Creating an Endpoint to receive the data
@app.post('/predict')
def predict(data : request_body):
    #making the data in a forl suitable for prediction
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predicting the Class 
    class_idx = clf.predict(test_data)[0]

    #Return the result
    return {'class' : iris.target_names[class_idx]}
