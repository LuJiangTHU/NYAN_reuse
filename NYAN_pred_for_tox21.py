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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

dir_path = os.path.dirname(os.path.realpath("__file__")) + "/"

sys.path.append("..")
from data import BondType

import logging

from tdc.single_pred import ADME

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

save = dir_path + "saves/ZINC-extmodel5hk-3M-325K" #loading the model we re-trained
batch_size = 1

task_name_list = os.listdir('./datasets/tox21_datasets/')


for task in task_name_list:

    print("Running tox21 task: ", task)

    dataset_name = task
    df = pd.read_csv('./datasets/tox21_datasets/{}'.format(dataset_name))

    data = np.array([x[0].split('\t') for x in df.values])[:,[1,4,2]].tolist()   #[id，smils, str]


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

            x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

            graph = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

            graph_data.append([graph, None])

            passed.append(smile)

        except Exception as error:

            print("Errored loading SMILES", smile)

    class EvalDataset (Dataset):

        def __init__ (self, **kwargs):
            super().__init__(**kwargs)

        def read (self):
            return [x[0] for x in graph_data]

    dataset = EvalDataset()
    loader = BatchLoader(dataset, batch_size=batch_size, epochs=1, mask=True, shuffle=False, node_level=False)

    predictions = encoder.predict(loader.load())
    predictions = [[float(y) for y in x] for x in predictions]

    all_data = list()

    for i in range(len(passed)):

        current_smiles = passed[i]

        appendable = [current_smiles] + predictions[i]
        all_data.append(appendable)    # [[id, smiles, label_str], latent vector]



    #Part 2

    #Training a model and performing cross-validation

    #shuffle the data again before cross-validation
    random.Random(18062022).shuffle(all_data)

    # Fivefold cross-validation, as per manuscript
    cross_validation_runs = 5

    results = {"AUROC": list(), "AUPRC": list(), "accuracy": list(),
               "f1": list(), "recall": list(), "mcc": list(), "gmean": list()}

    for i in range(cross_validation_runs):


        bottom = math.ceil(i/cross_validation_runs * len(all_data))
        top = math.ceil((i + 1)/cross_validation_runs * len(all_data))
        
        training_set = all_data[:bottom] + all_data[top:]
        testing_set = all_data[bottom:top]


        all_labels = sorted(list(set([int(x[0][2]) for x in all_data]))) #注意把str标签转化为int型

        training_latent_space = np.array([[float(y) for y in x[1:]] for x in training_set])
        training_labels = [all_labels.index(int(x[0][2])) for x in training_set]

        # Cap the latent spaces to [-1e8, 1e8]
        
        training_latent_space[training_latent_space == -np.inf] = -1e8
        training_latent_space[training_latent_space == np.inf] = 1e8

        clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                            ExtraTreesClassifier(n_estimators=512, max_features="log2", n_jobs=-1, random_state=0))
        clf.fit(training_latent_space, training_labels)


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

        gmean= geometric_mean_score(y_test[:, 0], np.round(y_score[:, 0]), average='binary')
        results["gmean"].append(gmean)

    # # Fivefold CV results
    # print(results)

    res = pd.DataFrame(results)
    res.to_csv('./result_reproduction_tox21/{}'.format(dataset_name))

    # Print the means and standard deviations

    auroc_mean = str(np.mean(results["AUROC"]))
    auroc_std = str(np.std(results["AUROC"]))

    auprc_mean = str(np.mean(results["AUPRC"]))
    auprc_std = str(np.std(results["AUPRC"]))

    accuracy_mean = str(np.mean(results["accuracy"]))
    accuracy_std = str(np.std(results["accuracy"]))

    print("--Running {} result--".format(task))
    print(f"Accuracy: {accuracy_mean} ± {accuracy_std}")
    print(f"AUROC: {auroc_mean} ± {auroc_std}")
    print(f"AUPRC: {auprc_mean} ± {auprc_std}")




    # Save a model
    # I'm going to take the last model from the CV and save it.
    # Of course, in practicable situations, you may want to fine-tune the exact
    # train and test set

    # os.makedirs("models", exist_ok=True)
    #
    # joblib.dump(clf, "models/" + dataset_name + ".joblib")
    # # %%
    # # Loading a model is as simple as instantiating the Joblib
    #
    # clf = joblib.load("models/" + dataset_name + ".joblib")


