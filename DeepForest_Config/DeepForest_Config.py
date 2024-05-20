#pip install deep-forest
import numpy as np
#sys.path.append("..") 
#from gcforest.gcforest import GCForest
from DeepForest_Config.gcforest.gcforest import GCForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV

#generate random dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

#train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def get_toy_config_LPIDF():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "LogisticRegression"} )
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 300, "max_depth": 5,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1,"num_class":2})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 300, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 300, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config

def get_config_ForSyn():
    config = {}
    ca_config = {}
    ca_config["random_state"] = None
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 2
    #config["if_stacking"]=False
    #config["if_save_model"]=False
    #config["train_evaluation"]=f1_macro ##f1_binary,f1_macro,f1_micro,accuracy
    ca_config["estimators"]=[]
    # for i in range(10):
    #     config["estimator_configs"].append({"n_fold":5,"type":"IMRF","n_estimators":40,"splitter":"best"})
    ca_config["estimators"].append({"n_folds":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    ca_config["estimators"].append({"n_folds":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    ca_config["estimators"].append({"n_folds":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    ca_config["estimators"].append({"n_folds":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    #config["output_layer_config"]=[]
    config["cascade"] = ca_config
    return config

def get_toy_config_HA():
    config = {}
    ca_config = {}
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 500,  "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "XGBClassifier", "n_estimators":500, "eval_metric": "auc", "objective": "binary:logistic", "nthread": -1})
    config["cascade"] = ca_config
    return config

def get_config_deepExtraTree_500x3():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0    
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1}) 
    config["cascade"] = ca_config
    return config

def get_config_deepExtraTree():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0    
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})    
    config["cascade"] = ca_config
    return config
#build the deep forest classification

config = get_config_deepExtraTree()

forest = GCForest(config)

#train the model
forest.fit_transform(X_train, y_train)

#make prediction
y_pred = forest.predict(X_test)

#evaluation
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)