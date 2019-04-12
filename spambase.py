import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB



#read data file
data = pd.read_csv("spambase.data", header=None)
data.rename(columns={57:'class'}, inplace=True)
y = np.array(data.pop('class'))
X = np.array(data)


#cross validation
kf = KFold(n_splits=5, shuffle=True)

# choose model from three types naive_bayes classifier
# clf = GaussianNB()
# clf = clf = MultinomialNB()
clf = BernoulliNB()
Scores = []
acc = []
total = 0


for train_index, test_index in kf.split(X):
    #  split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # fit model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    # compute fpr, fnr, accuracy,error
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (tp + tn + fn + fp)
    fnr = fn / (tp + tn + fn + fp)
    total = tp + tn + fn + fp
    acc.append(score)
    Scores.append({"fp":fp,"fn":fn,"total":total,"false positive(%)":fpr * 100,
                   "false negative(%)":fnr * 100,"accuracy(%)": score * 100, "error(%)": 100 - score*100})

# print each fold's accuracy and error
df = pd.DataFrame(Scores, columns=['fp', 'fn','total',
                                   'false positive(%)','false negative(%)','accuracy(%)', 'error(%)'])
print(df)
print("Average accuracy: ", np.mean(acc) * 100,"%")
print("Average Error:", 100 - np.mean(acc) * 100,"%")


