#pip install deep-forest
import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV


#generate random dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

#train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build the deep forest classification
forest = CascadeForestClassifier(n_estimators=10, n_trees=100, max_depth=5, random_state=42)

#train the model
forest.fit(X_train, y_train)

#make prediction
y_pred = forest.predict(X_test)

#evaluation
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

