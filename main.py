import torch
import numpy as np
import pickle as pkl
import os

def get_cell_list(data_dir, seq_len):
    cell_list = []
    for item in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, item)):
            cell = os.path.join(data_dir, item)
            with open(cell, 'rb') as f:
                data = pkl.load(f)
                if data['observation'].shape[0] >= seq_len:
                    cell_list.append(cell)
    return cell_list

def main():
    cell_list = get_cell_list('processed_data', 100)
    assert len(cell_list) == 92
    for cell in cell_list:
        datas = []
        with open(cell, 'rb') as f:
            data  = pkl.load(f)
            datas.append(data['action'])
    datas = np.concatenate(datas, axis=0)
    print(np.mean(datas, axis=0), np.max(datas, axis=0) - np.min(datas, axis=0))



if __name__ == '__main__':
    main()