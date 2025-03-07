# import pandas
import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load dataset
pima = pd.read_csv('diabetes.csv', skiprows=1, names=col_names)

print(pima.head())

# split dataset in features and target variables

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

X = pima[feature_cols]
y = pima.label

# split X and y into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameter)
logreg = LogisticRegression(random_state=16, max_iter=500)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)


# Model evaluation using confusion matrix

# import the metrics class
from sklearn import metrics

cnf_matrics = metrics.confusion_matrix(y_test, y_pred)
cnf_matrics

# visualizing confusion matrics

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

class_names = [0, 1] # names of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrics), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Pedicted label')

plt.show()

# confusion matrix evaluation metrics
from sklearn.metrics import classification_report
target_names = ['Without diabetes', 'With diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))


# ROC curve (Receiver Operating Characteristic)

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()