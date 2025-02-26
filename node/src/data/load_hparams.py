import pickle as pkl
from os.path import join as pjoin, exists as pexists
import json
    

def load_hparams(data_name: str, data_path = './hparams/node/'):
    path = pjoin(data_path + data_name + '.json')
    with open(path, 'rb') as f_in:
        return json.load(f_in)