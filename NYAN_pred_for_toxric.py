import os
import sys
import math
import warnings
import json
import csv
import random
import loading
import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from DeepForest_Config.gcforest.gcforest import GCForest
from DeepForest_Config.DeepForest_Config import get_toy_config_HA, get_config_ForSyn, get_config_deepExtraTree
from sklearn.tree import DecisionTreeClassifier


dir_path = os.path.dirname(os.path.realpath("__file__")) + "/"

sys.path.append("..")
from data import BondType

import logging

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from spektral.data import BatchLoader, Dataset, Graph
from spektral import transforms

from model import VAE, NyanEncoder, NyanDecoder, EpochCounter

import scipy as sp
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
else:
    print('No GPU available')


logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

strategy = tf.distribute.MirroredStrategy()

save = dir_path + "saves/ZINC-extmodel5hk-3M"  # loading the original well-trained model
batch_size = 1

task_name_list = os.listdir('./datasets/toxric_datasets/')  #obtain the task list


for task in task_name_list:  # run on each task dataset

    print("Running toxric task: ", task)
    dataset_name = task

    # read the dataset
    df = pd.read_csv('./datasets/toxric_datasets/{}'.format(dataset_name))
    data = df[['TAID', 'Canonical SMILES', 'Toxicity Value']]
    data = data.dropna(subset=['Canonical SMILES'])
    data = data.values.tolist()

    with strategy.scope():

        encoder = NyanEncoder(latent_dim=64, batched=True)
        decoder = NyanDecoder(fingerprint_bits=679, regression=1613)

        model = VAE(encoder, decoder)
        model.load_weights(save)

    print("Generating latents using the save {}".format(save))

    all_smiles = data

    # Initialise dataset
    graph_data = list()
    passed = list()

    print("Loading {} molecules".format(len(all_smiles)))

    for smile in tqdm(all_smiles):

        if smile[1] == "":
            continue

        try:

            graph = loading.get_data(smile[1], only_biggest=True, unknown_atom_is_dummy=True)

            x, a, e = loading.convert(*graph,
                                      bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC,
                                             BondType.NOT_CONNECTED])

            graph = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

            graph_data.append([graph, None])

            passed.append(smile)

        except Exception as error:

            print("Errored loading SMILES", smile)


    class EvalDataset(Dataset):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def read(self):
            return [x[0] for x in graph_data]


    dataset = EvalDataset()
    loader = BatchLoader(dataset, batch_size=batch_size, epochs=1, mask=True, shuffle=False, node_level=False)

    predictions = encoder.predict(loader.load())
    predictions = [[float(y) for y in x] for x in predictions]

    all_data = list()

    for i in range(len(passed)):
        current_smiles = passed[i]

        appendable = [current_smiles] + predictions[i]
        all_data.append(appendable)    # [[id, smiles, label], latent vector]



    # shuffle the data again before cross-validation
    random.Random(18062022).shuffle(all_data)

    # Fivefold cross-validation, as per manuscript
    cross_validation_runs = 5

    models = ['ExtraTree', 'RF', 'LGB', 'XGB', 'GBDT', 'Adaboost', 'SVM', 'DF']

    for model in models:  # run using different learning algorithm

        save_path = os.path.join(dir_path, 'result_toxric', model)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if model == 'ExtraTree':
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                ExtraTreesClassifier(n_estimators=512, max_features="log2", n_jobs=-1, random_state=0))
        elif model == 'RF':
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                RandomForestClassifier(n_estimators=50, max_depth=10))
        elif model == 'LGB':
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                lgb.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05,
                                                   n_estimators=20))
        elif model == 'XGB':
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                xgb.XGBClassifier(eta=0.1, max_depth=8, min_child_weight=2, gamma=10, subsample=0.85,
                                                  colsample_bytree=0.8))

        elif model == 'GBDT':
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                           random_state=42))
        elif model == 'Adaboost':
            base_classifier = DecisionTreeClassifier(max_depth=5)
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                AdaBoostClassifier(base_classifier, n_estimators=50, learning_rate=1.0,
                                                   random_state=42))
        elif model == 'SVM':
            clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                                svm.SVC(C=9, kernel='rbf', gamma=0.01, decision_function_shape='ovo', probability=True))

        # deep forest model
        elif model == 'DF':
            config = get_config_deepExtraTree()
            clf = GCForest(config)
        # elif model == 'DNN':
        #     clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), DNN(2))


        # Training a model and performing cross-validation


        results = {"AUROC": list(), "AUPRC": list(), "accuracy": list(),
                   "f1": list(), "recall": list(), "mcc": list(), "gmean": list()}


        for i in range(cross_validation_runs):  # 5 CV

            bottom = math.ceil(i / cross_validation_runs * len(all_data))
            top = math.ceil((i + 1) / cross_validation_runs * len(all_data))

            training_set = all_data[:bottom] + all_data[top:]
            testing_set = all_data[bottom:top]

            all_labels = sorted(list(set([int(x[0][2]) for x in all_data])))

            training_latent_space = np.array([[float(y) for y in x[1:]] for x in training_set])
            training_labels = [all_labels.index(int(x[0][2])) for x in training_set]

            # Cap the latent spaces to [-1e8, 1e8]

            training_latent_space[training_latent_space == -np.inf] = -1e8
            training_latent_space[training_latent_space == np.inf] = 1e8

            if model == 'DF':
                clf.fit_transform(training_latent_space, np.array(training_labels))
            else:
                clf.fit(training_latent_space, training_labels)
            #
            #
            # config = get_toy_config_HA()
            # clf = GCForest(config)
            # clf.fit_transform(training_latent_space, np.array(training_labels))

            testing_latent_space = np.array([[float(y) for y in x[1:]] for x in testing_set])
            testing_labels = [all_labels.index(int(x[0][2])) for x in testing_set]

            # Test the model
            y_score = clf.predict_proba(testing_latent_space)
            y_test = testing_labels


            # Convert y_test to one-hot
            def create_one_hot(size, index):

                vector = [0] * size
                vector[index] = 1

                return vector


            y_test = np.array([create_one_hot(len(all_labels), x) for x in y_test])

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(len(all_labels)):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            results["AUROC"].append(roc_auc[0])

            # Strictly not equivalent, but more pessimistic
            auprc = metrics.average_precision_score(y_test[:, 0], y_score[:, 0])
            results["AUPRC"].append(auprc)

            accuracy = metrics.accuracy_score(y_test[:, 0], np.round(y_score[:, 0]))
            results["accuracy"].append(accuracy)

            f1 = f1_score(y_test[:, 0], np.round(y_score[:, 0]), average='binary')
            results["f1"].append(f1)

            recall = recall_score(y_test[:, 0], np.round(y_score[:, 0]), average="binary")
            results["recall"].append(recall)

            mcc = matthews_corrcoef(y_test[:, 0], np.round(y_score[:, 0]))
            results["mcc"].append(mcc)

            gmean = geometric_mean_score(y_test[:, 0], np.round(y_score[:, 0]), average='binary')
            results["gmean"].append(gmean)

        # Fivefold CV results
        res = pd.DataFrame(results)
        res.to_csv('./result_toxric/{}/{}'.format(model, dataset_name))

        # Print the means and standard deviations

        auroc_mean = str(np.mean(results["AUROC"]))
        auroc_std = str(np.std(results["AUROC"]))

        auprc_mean = str(np.mean(results["AUPRC"]))
        auprc_std = str(np.std(results["AUPRC"]))

        accuracy_mean = str(np.mean(results["accuracy"]))
        accuracy_std = str(np.std(results["accuracy"]))

        print("--Running {} result using {} --".format(task, model))
        print(f"Accuracy: {accuracy_mean} ± {accuracy_std}")
        print(f"AUROC: {auroc_mean} ± {auroc_std}")
        print(f"AUPRC: {auprc_mean} ± {auprc_std}")

    # Part 3

    # Saving a model to Joblib so that it can be retrieved again later

    # Save a model
    # I'm going to take the last model from the CV and save it.
    # Of course, in practicable situations, you may want to fine-tune the exact
    # train and test set

    # os.makedirs("models", exist_ok=True)
    #
    # joblib.dump(clf, "models/" + dataset_name + ".joblib")
    #
    # # Loading a model is as simple as instantiating the Joblib
    #
    # clf = joblib.load("models/" + dataset_name + ".joblib")


