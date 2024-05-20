import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

datapoints = open("datasets/mordred.tsv").read().split("\n")
# headers = open("datasets/mordred_headers.txt").read().split("\n")
fingerprints = open("datasets/maccs_morgan.tsv").read().split("\n")

writable = list()


for i in tqdm(range(len(fingerprints))):

    if fingerprints[i] == "":
        continue

    fingerprint = fingerprints[i].split("\t")
    mordred = datapoints[i].split("\t")

    if fingerprint[0] != mordred[0]:
        continue

    # print(mordred[0] + "\t" + fingerprint[1] + "," + mordred[1])

    writable.append(mordred[0] + "\t" + fingerprint[1] + "," + mordred[1])

open('datasets/3m_512.tsv', "w+").write('\n'.join([x for x in writable]))

exit()
