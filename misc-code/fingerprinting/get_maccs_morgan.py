from rdkit import Chem
from rdkit.Chem import MACCSkeys,AllChem
from tqdm import tqdm

smiles = open("datasets/centres.smi").read().split("\n")

writable = list()

for smile_entry in tqdm(smiles):

	current_smiles = smile_entry.split("\t")[0]
	mol = Chem.MolFromSmiles(current_smiles)

	fp_maccs = MACCSkeys.GenMACCSKeys(mol)
	fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=512, useBondTypes=True)
	writable.append(current_smiles + '\t' + fp_maccs.ToBitString() +fp_morgan.ToBitString() )


open('datasets/maccs_morgan.tsv', "w+").write('\n'.join([x for x in writable]))



