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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics

dir_path = os.path.dirname(os.path.realpath("__file__")) + "/"
print(dir_path)

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

save = dir_path + "saves/ZINC-extmodel5hk-3M-325K"  #loading the model we re-trained
batch_size = 1


# Part 1
# Please make sure that you have your Conda environment previously instantiated!
# I will use the dataset BBB_Martins as an example

task_list = ["BBB_Martins","Pgp_Broccatelli","CYP2D6_Veith"]

for task in task_list:
    print("Running ADMET task: ", task)

    dataset_name = task

    data = ADME(name=dataset_name)


    # Please take note! I are using random split here instead of scaffold split.
    # You can refer to Supplementary File 1 for scaffold split results.
    # The reason why random split is done here is for fivefold cross-validation.
    split = data.get_split(method = "random")

    # Combine all the data together in this case since I'm doing CV
    data = split["train"].values.tolist() + split["valid"].values.tolist() + split["test"].values.tolist()


    with strategy.scope():

        encoder = NyanEncoder(latent_dim=64, batched=True)
        decoder = NyanDecoder(fingerprint_bits=679, regression=1613)

        model = VAE(encoder, decoder)
        model.load_weights(save).expect_partial()

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
    # for i in range(100):
        current_smiles = passed[i]

        appendable = [current_smiles] + predictions[i]
        all_data.append(appendable)  # [[id, smiles, label_str], latent vector]


    # Usually I shuffle the data again before cross-validation, just in case
    # Don't want to introduce biases unknowingly.

    random.Random(18062022).shuffle(all_data)

    # Fivefold cross-validation, as per manuscript
    cross_validation_runs = 5

    results = {"AUROC": list(), "AUPRC": list(), "accuracy": list()}

    for i in range(cross_validation_runs):

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

        # This is the part where you decide what kind of model you want to use
        # In this case, I'll follow what we originally did in the manuscript for the BBB_Martins dataset
        # Which is to use ExtraTreesClassifiers
        # You can use other pipelines of interest, even autoML methods

        clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True),
                            ExtraTreesClassifier(n_estimators=512, max_features="log2", n_jobs=-1, random_state=0))
        clf.fit(training_latent_space, training_labels)

        testing_latent_space = np.array([[float(y) for y in x[1:]] for x in testing_set])
        testing_labels = [all_labels.index(int(x[0][2])) for x in testing_set]

        # print(len(testing_labels))

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

    # Fivefold CV results
    print(results)

    res = pd.DataFrame(results)
    res.to_csv('./result_reproduction_ADMET/{}.csv'.format(dataset_name))


    # Print the means and standard deviations

    auroc_mean = str(np.mean(results["AUROC"]))
    auroc_std = str(np.std(results["AUROC"]))

    auprc_mean = str(np.mean(results["AUPRC"]))
    auprc_std = str(np.std(results["AUPRC"]))

    accuracy_mean = str(np.mean(results["accuracy"]))
    accuracy_std = str(np.std(results["accuracy"]))

    print(f"Accuracy: {accuracy_mean} ± {accuracy_std}")
    print(f"AUROC: {auroc_mean} ± {auroc_std}")
    print(f"AUPRC: {auprc_mean} ± {auprc_std}")
