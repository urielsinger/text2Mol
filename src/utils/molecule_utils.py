import numpy as np

import urllib

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg

model = Word2Vec.load(r"..\dump\model_300dim.pkl")

def smile2molecule(smile):
    return Chem.MolFromSmiles(smile)

def molecule2sentence(molecule, radius=1):
    sentence = mol2alt_sentence(molecule, radius=radius)
    return sentence

def smile2sentence(smile, radius=1):
    molecule = smile2molecule(smile)
    sentence = molecule2sentence(molecule, radius=radius)
    return sentence

def sentence2vec(sentence, mode='vec', unseen='UNK'):
    if mode == 'vec':
        return sentences2vec([sentence], model, unseen='UNK')[0]
    else:
        keys = set(model.wv.vocab.keys())

        if unseen:
            unseen_vec = model.wv.word_vec(unseen)
            x = [model.wv.word_vec(y) if y in set(sentence) & keys
                 else unseen_vec for y in sentence]
        else:
            x = [model.wv.word_vec(y) for y in sentence
                 if y in set(sentence) & keys]

        return np.array(x)

def smile2vec(smile, radius=1, mode='vec', unseen='UNK'):
    return sentence2vec(smile2sentence(smile, radius=radius), mode=mode, unseen=unseen)

def molecule2vec(molecule, radius=1, mode='vec', unseen='UNK'):
    return sentence2vec(molecule2sentence(molecule, radius=radius), mode=mode, unseen=unseen)

def name2smile(name):
    url = f'https://cactus.nci.nih.gov/chemical/structure/{name}/smiles'
    smile = urllib.request.urlopen(url).read()
    return smile

