import numpy as np
import  urllib.request

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn import metrics
from sklearn.metrics import accuracy_score



f1 = open("train1.txt", "r")
train_data = f1.readlines()
f1.close()


dataset1 = np.loadtxt(train_data, delimiter = ',')

x_train = dataset1[:, 0:784]
y_train = dataset1[:, 784]


f2 = open("test1.txt", "r")
test_data = f2.readlines()
f2.close()


dataset2 = np.loadtxt(test_data, delimiter = ',')


x_test = dataset2[:, 0:784]
y_test = dataset2[:, 784]


# Naive Bayes ==========================================================
Multi = MultinomialNB()
Multi.fit(x_train, y_train)
y_pred = Multi.predict(x_test)

print("Naive Bayes : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")



# Random Forest ==========================================================

print("\n")
print("\n")

trees = 0

while trees<40:
    trees+=5

    RandFor = RandomForestClassifier(n_estimators=trees)
    RandFor.fit(x_train, y_train)
    y_pred = RandFor.predict(x_test)
    
    print("Random Forset (trees = " + str(trees)+" ) : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")
