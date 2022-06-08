import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skms
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


#seed for reproducable results
np.random.seed(10)

#these functions were some the pre-processing/cleaning options experimented with.
def meanReplacement(data):
    #repacing the 0 values with the mean of the feature
    cols = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"]
    for col in cols:
        data[col] = data[col].replace(0, data[data[col] != 0][col].mean())
    
    #tried with and without including Insulin
    """cols = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin", "DiabetesPedigreeFunction", "Age"]
    for col in cols:
        data[col] = data[col].replace(0, data[data[col] != 0][col].mean())"""
    
    
    return(data)

def meanClassReplacement(data):
    #replacing 0 values with the mean of the class they belong to
    negativeClassMean = {}
    positiveClassMean = {}

    cols = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age", "Insulin"]
    for col in cols:
        negativeClassMean[col] = data[(data['Outcome']==0) & (data[col]!=0)][col].mean()
        positiveClassMean[col] = data[(data['Outcome']==1) & (data[col]!=0)][col].mean()

    for col in negativeClassMean:
        data.loc[(data[col]==0) & (data['Outcome'] == 0), col] = negativeClassMean[col]
        data.loc[(data[col]==0) & (data['Outcome'] == 1), col] = positiveClassMean[col]

    return(data)

def normalising(data):
    #transforming the data to a standard normal distribution
    cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    data[cols_to_norm] = StandardScaler().fit_transform(data[cols_to_norm])

    return(data)

def overSampling(rawData, labels):
    #creating synthetic data to create a balanced dataset
    sampler = RandomOverSampler(sampling_strategy='minority')
    rawData, labels = sampler.fit_resample(rawData, labels)

    return(rawData, labels)

#import the data as a dataFrame
fpath = 'diabetes.csv'
data = pd.read_csv(fpath)

#the best approaches for pre-processing/cleaning for KNN
data = meanClassReplacement(data)
data = normalising(data)

#split the labels from the data
rawData = data.filter(["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
labels = data.filter(["Outcome"])

#oversample the data
rawData, labels = overSampling(rawData, labels)

#split the data (and their labels) into training and testing batches
trainData, testData, trainLabels, testLabels = skms.train_test_split(rawData, labels, train_size=0.8, test_size=0.2, random_state=10, stratify=labels)
trainLabels = np.ravel(trainLabels)

#conduct a search the best possible parameters for KNN
#this parameters dictionairy is (nearly) all possible parameters and the model becomes prone to overfitting if used, 
#hence why trial and error was used to identify better parameters to pass through the search.
"""params = [{'n_neighbors': [n for n in range(1,31,2)], 'weights': ['uniform', 'distance'],
         'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size': [n for n in range(1,50)], 'p' : [1, 2, 3, 4, 5],
         'metric' : ['euclidean', 'manhattan']}]"""

#after experimenting - this parameters dictionairy produces the best results
params = [{'n_neighbors': [n for n in range(1,31,2)], 'weights': ['uniform', 'distance'],'p' : [1, 2, 3, 4, 5],
         'metric' : ['euclidean', 'manhattan']}]

knn = GridSearchCV(KNeighborsClassifier(), params, scoring='accuracy', cv=5)
knn.fit(trainData, trainLabels)
print("The best parameters are:", knn.best_params_)
print("The accuracy score on the test data is:", knn.score(testData, testLabels))

#visualise the results in a confusion matrix
predictedLabels = knn.predict(testData)
confusionMatrix = confusion_matrix(testLabels, predictedLabels, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=knn.classes_)
disp.plot()
plt.show()


