"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/12/8 2:06
@File : dataloader.py
"""
import time

from init import *


class TrainDataset(Dataset):
    def __init__(self, df_dict, seq_len=60, target='Target5'):
        self.BTC = df_dict['BTC'][['Open', 'High', 'Low', 'Close', 'Volume', target]]
        self.ETH = df_dict['ETH'][['Open', 'High', 'Low', 'Close', 'Volume', target]]
        self.LTC = df_dict['LTC'][['Open', 'High', 'Low', 'Close', 'Volume', target]]
        self.XRP = df_dict['XRP'][['Open', 'High', 'Low', 'Close', 'Volume', target]]

        self.seq_len = seq_len
        self.btc_len = len(self.BTC) - self.seq_len - 1
        self.eth_len = len(self.ETH) - self.seq_len - 1
        self.ltc_len = len(self.LTC) - self.seq_len - 1
        self.xrp_len = len(self.XRP) - self.seq_len - 1

    def __len__(self):
        # return len(self.BTC) + len(self.ETH) + len(self.LTC) + len(self.XRP) - 4 * self.seq_len
        return self.btc_len + self.eth_len + self.ltc_len + self.xrp_len

    def __getitem__(self, item):
        if item <= self.btc_len:
            return (torch.Tensor(self.BTC.iloc[item: item + self.seq_len, 0:5].values),
                    self.BTC.iloc[item + self.seq_len, 5])
        elif item - self.btc_len <= self.eth_len:
            item = item - self.btc_len
            return (torch.Tensor(self.ETH.iloc[item: item + self.seq_len, 0:5].values),
                    self.ETH.iloc[item + self.seq_len, 5])
        elif item - self.btc_len - self.eth_len <= self.ltc_len:
            item = item - self.btc_len - self.eth_len
            return (torch.Tensor(self.LTC.iloc[item: item + self.seq_len, 0:5].values),
                    self.LTC.iloc[item + self.seq_len, 5])
        else:
            item = item - self.btc_len - self.eth_len - self.ltc_len
            return (torch.Tensor(self.XRP.iloc[item: item + self.seq_len, 0:5].values),
                    self.XRP.iloc[item + self.seq_len, 5])


def get_df_dict(data_path=r'.\data', train_=True):
    if train_:
        f_l = [os.path.join(data_path, f) for f in os.listdir(data_path) if 'train.pkl' in f]
    else:
        f_l = [os.path.join(data_path, f) for f in os.listdir(data_path) if 'valid.pkl' in f]

    df_d = {}
    for f in f_l:
        df_ = pd.read_pickle(f)
        name_ = re.findall(r'[A-Z]{3}', f)[0]
        df_d[name_] = df_
    return df_d


def get_train_valid_loader(config, shuff=True):
    tr_dict = get_df_dict(config['PATH']['DATA'], train_=True)
    va_dict = get_df_dict(config['PATH']['DATA'], train_=False)

    tr_dataset_ = TrainDataset(tr_dict,
                               config['SEQ_LEN'],
                               config['TARGET'])
    va_dataset_ = TrainDataset(va_dict,
                               config['SEQ_LEN'],
                               config['TARGET'])

    tr_loader_ = DataLoader(tr_dataset_,
                            batch_size=config['BATCH_SIZE'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=shuff)
    va_loader_ = DataLoader(va_dataset_,
                            batch_size=config['BATCH_SIZE'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=shuff)

    return tr_loader_, va_loader_


class StoreAllDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def generate_dataloader_all(config):
    print('================ Generating Data ================')
    tr_dict = get_df_dict(config['PATH']['DATA'], train_=True)
    va_dict = get_df_dict(config['PATH']['DATA'], train_=False)

    tr_dataset_ = TrainDataset(tr_dict,
                               config['SEQ_LEN'],
                               config['TARGET'])
    va_dataset_ = TrainDataset(va_dict,
                               config['SEQ_LEN'],
                               config['TARGET'])
    tr_list = []
    va_list = []
    time0 = time.time()
    for i in range(len(tr_dataset_)):
        tr_list.append(tr_dataset_[i])
    time1 = time.time()
    for i in range(len(va_dataset_)):
        va_list.append(va_dataset_[i])
    print(f'Data generation finish:= Train:{round(time1 - time0, 2)} Valid:{round(time.time() - time1, 2)}s')

    tr_dataset_new = StoreAllDataset(tr_list)
    va_dataset_new = StoreAllDataset(va_list)

    tr_loader_ = DataLoader(tr_dataset_new,
                            batch_size=config['BATCH_SIZE'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)
    va_loader_ = DataLoader(va_dataset_new,
                            batch_size=config['BATCH_SIZE'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)
    return tr_loader_, va_loader_


def generate_dataloader_load(config):
    print('================ Generating Data ================')
    tr_path = os.path.join(config['PATH']['DATASET'], 'train_dataset.pth')
    va_path = os.path.join(config['PATH']['DATASET'], 'valid_dataset.pth')

    tr_dataset_load = torch.load(tr_path, map_location=torch.device('cpu'))
    va_dataset_load = torch.load(va_path, map_location=torch.device('cpu'))

    tr_loader_ = DataLoader(tr_dataset_load,
                            batch_size=config['BATCH_SIZE'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)
    va_loader_ = DataLoader(va_dataset_load,
                            batch_size=config['BATCH_SIZE'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)

    return tr_loader_, va_loader_
