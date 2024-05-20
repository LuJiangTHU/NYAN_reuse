#pip install deep-forest
import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#generate random dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

#train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### refine the model### #
#define the grid
param_grid = {
    'n_estimators': [10,20,50,100],
    'n_trees': [10,20,50,100,200],
    'max_depth': [3,5,7,10,15]
}

#bulid the model
forest = CascadeForestClassifier(random_state=42)

#refine 
grid_search = GridSearchCV(forest, param_grid, cv=5)
grid_search.fit(X_train, y_train)

#output the bset parameters and the accuracy
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)
