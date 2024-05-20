import os
import sys
import math
import warnings
import csv
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from DeepForest_Config.gcforest.gcforest import GCForest
from DeepForest_Config.DeepForest_Config import get_toy_config_HA, get_config_ForSyn, get_config_deepExtraTree
from sklearn.tree import DecisionTreeClassifier
import logging

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=True)


dir_path = os.path.dirname(os.path.realpath("__file__")) + "/"

sys.path.append("..")

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

batch_size = 1

task_name_list = os.listdir('./datasets/toxric_datasets/')  #obtain the task list

#['MACCS', 'Morgan512','Morgan1024','AtomPair', 'Topological Torsion', 'Avalon', 'Rdkit2D', 'Mordred', 'ECFP2']
otherFG = 'Avalon'

# df_fg = pd.read_csv('./datasets/otherFGs_for_toxric/{}.csv'.format(otherFG))


for task in task_name_list:  # run on each task dataset

    print("Running toxric task: ", task)
    dataset_name = task

    # read the dataset
    df = pd.read_csv('./datasets/toxric_datasets/{}'.format(dataset_name))
    df = df[['TAID', 'Canonical SMILES', 'Toxicity Value']]
    df = df.dropna(subset=['Canonical SMILES'])
    data = df.values.tolist()

    print("Loading {} molecules".format(len(df)))


    if otherFG in ['MACCS', 'Morgan512', 'Morgan1024', 'Atompair','Topological Torsion', 'Avalon', 'Mordred']:

        all_data = []

        for x in tqdm(data):
            smile = x[1]  # SMILE
            mol = Chem.MolFromSmiles(smile)

            if otherFG == 'MACCS':    #167-d
                fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
                all_data.append(x + [int(bit) for bit in fp])

            elif otherFG=='Morgan512':   #512-d
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=512, useBondTypes=True).ToBitString()
                all_data.append(x + [int(bit) for bit in fp])

            elif otherFG=='Morgan1024':   #1024-d
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024, useBondTypes=True).ToBitString()
                all_data.append(x + [int(bit) for bit in fp])

            elif otherFG=='Atompair':  #2048-d  ->1024
                fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024).ToBitString()
                all_data.append(x + [int(bit) for bit in fp])

            elif otherFG=='Topological Torsion':  #2048-d >1024
                fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024).ToBitString()
                all_data.append(x + [int(bit) for bit in fp])

            elif otherFG=='Avalon':  #512-d -> 1024-d
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=1024).ToBitString()
                all_data.append(x + [int(bit) for bit in fp])

            elif otherFG=='Mordred':  #1613-d
                descriptors = calc(mol)
                raw_descriptors = descriptors.values()
                fp = list()

                for raw_descriptor in raw_descriptors:
                    if type(raw_descriptor) is float or type(raw_descriptor) is int:
                        fp.append(str(raw_descriptor))
                    else:
                        fp.append("0")

                all_data.append(x + fp)


    else:

        df_fg = pd.read_csv('./datasets/otherFGs_for_toxric/{}.csv'.format(otherFG))

        all_data = pd.merge(left=df, right=df_fg).values.tolist()


    # shuffle the data again before cross-validation
    random.Random(18062022).shuffle(all_data)

    # Fivefold cross-validation, as per manuscript
    cross_validation_runs = 5

    # models = ['ExtraTree', 'RF', 'LGB', 'XGB', 'GBDT', 'Adaboost', 'SVM', 'DF']
    models = ['ExtraTree', 'RF', 'XGB', 'SVM', 'DF']  # select several strong methods

    for model in models:  # run using different learning algorithm

        save_path = os.path.join(dir_path, 'result_otherFG_toxric', otherFG, model)

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

            all_labels = sorted(list(set([int(x[2]) for x in all_data])))

            training_latent_space = np.array([[float(y) for y in x[3:]] for x in training_set])
            training_labels = [all_labels.index(int(x[2])) for x in training_set]

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

            testing_latent_space = np.array([[float(y) for y in x[3:]] for x in testing_set])
            testing_labels = [all_labels.index(int(x[2])) for x in testing_set]

            # Test the model
            y_score = clf.predict_proba(testing_latent_space)  #[num, 2]
            y_test = testing_labels  # list type


            # Convert y_test to one-hot
            def create_one_hot(size, index):

                vector = [0] * size
                vector[index] = 1

                return vector


            y_test = np.array([create_one_hot(len(all_labels), x) for x in y_test]) # [num, 2]

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
        res.to_csv('./result_otherFG_toxric/{}/{}/{}'.format(otherFG, model, dataset_name))

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




