## About
**NotYetAnotherNightshade** (NYAN) is a graph variational encoder as described in the article "Application of variational graph encoders as an effective generalist algorithm in holistic computer-aided drug design" published in Nature Machine Intelligence, 2023. In NYAN, the low-dimension latent variables derived from the variational graph autoencoder are leveraged as a kind of universal molecular representation, yielding remarkable performance and versatility throughout the drug discovery process.

We assess the reusability of NYAN and comprehensively investigate its applicability within the context of specific chemical toxicity prediction. We used more expanded predictive toxicology datasets sourced from TOXRIC, a comprehensive and standardized toxicology database (<span style="color:red;">Lianlian Wu, Bowei Yan, Junshan Han, Ruijiang Li, Jian Xiao, Song He, Xiaochen Bo. TOXRIC: a comprehensive database of toxicological data and benchmarks, Nucleic Acids Research, 2022, https://toxric.bioinforai.tech/home</span>).

For toxicity prediction tasks, we compiled 30 assay endpoints related to toxic effects, organ toxicity, and clinical toxicity. In the case of acute toxicity prediction, the dataset includes 59 endpoints with 80,081 unique compounds and 122,594 measurements.

Across these professional toxicity datasets, the toxicity prediction performance via NYAN latent representation and other popular molecular feature representations are experimentally benchmarked, and the adaptation of the NYAN latent representation to other downstream surrogate models are also explored.

Furthermore, we integrate the variational graph encoder of NYAN with multi-task learning paradigm to boost the multi-endpoint acute toxicity prediction. The code of this part can be found in our another code repository: https://github.com/LuJiangTHU/Acute_Toxicity_NYAN.git

This repository contains the code we used in reproducing the original results in ADMET and Tox21 experiments, benchmarking the performance of different molecular representation methods and the exploration of competent surrogate models in toxicity prediction based on the TOXRIC database. 



## Installation
```sh
git clone https://github.com/LuJiangTHU/NYAN_reuse.git
cd NYAN_reuse
```

```sh
conda env create -f environment.yml
conda activate nyan
```


## Reproduction
In this code repository, most of core codes were directly downloaded from the code repository provided by the authors of original article (https://github.com/Chokyotager/NotYetAnotherNightshade.git). Original article used 650K molecular data from ZINC database to train their framework and then obtained a model named `ZINC-extmodel5hk-3M`. In contrast, we only used half of training data of original paper (325K molecules versus original 650K, and 325K is a subset of 650K) to retrain the NYAN framework and then obtained another model `ZINC-extmodel5hk-3M-325K`. Our reproduction experiments were based on this retrained NYAN model.

### Obtaining the training data and retraining NYAN
The `/datasets/centres.smi` contains 700K molecular SMILEs. Original article used the anterior 650K as its training data, while we used the anterior 325K SMILEs. You can sequentially use the 3 scripts, including `get_maccs_morgan.py`, `get_mordred.py`, and `make_3m.py` in the folder of `/misc-code/fingerprinting/`, to obtain the combined training set named `'datasets/3m_512.tsv'`.

The `config.json` can be used to control the training configurations including the number of training data. Using the following command to retrain your own NYAN framework:
```sh
python train.py
```
 
### Prediction on ADMET
Using the following command to preform the ADMET prediction:
```sh
python NYAN_pred_for_ADMET.py
```

The output results will be saved into `/results_reproduction_ADMET/`.

### Prediction on Tox21
Using the following command to preform the Tox21 prediction:
```sh
python NYAN_pred_for_tox21.py
```

The output results will be saved into the folder of `/result_reproduction_tox21/`.

 
## Benchmarking of different molecular representation methods
Please use "otherFG_pred_for_toxric.py" to run on the 30 datasets from TOXRIC database with different molecular features inlcluding Rdkit2D, Mordred, Avalon, Atom pair, Morgan512, Morgan1024, Topological Torsion, MACCS and ECFP2:
```sh
python otherFG_pred_for_toxric.py
```
The output will be tab-delimited and the detailed results w.r.t different molecular representation will be saved into the folder of `/result_otherFG_toxric/`. 

## Exploring different surrogate models
Please use "NYAN_pred_for_toxric.py" to run on the 30 datasets from TOXRIC database with other popular toxicity classification algorithm including Extra Tree, Deep Forest, Support Vector Machine (SVM), Random Forest (RF), Adaboost, Light GBM (LGB), gradient-boosted decision tree (GBDT), and Xgboost (XGB):
```sh
python NYAN_pred_for_toxric.py
```
The output will be tab-delimited and the detailed results w.r.t different surrogate models will be saved into the folder of `/result_toxric/`.

## Deriving the NYAN latent representations for acute toxicity data
Please use "encoder_59endpoints_smiles.py" to derive the 64-dimension NYAN latent representation for the 80081 chemical compounds in acute toxicity dataset:
```sh
python encoder_59endpoints_smiles.py
```
The generated NYAN latent representations will be tab-delimited e saved into the folder of `/datasets/MTL/`.

## Enhancing the multi-endpoint acute toxicity prediction using NYAN
For the multi-task learning experiments on acute toxicity prediction, we firstly used the re-trained NYAN 325K model to derive the NYAN latent representations for the chemical compounds in acute toxicity dataset (see the previous section), and then transfer these NYAN latent representations to our another code project ('Acute_Toxicity_NYAN', see https://github.com/LuJiangTHU/Acute_Toxicity_NYAN.git) to perform multi-endpoint acute toxicity prediction experiments.



