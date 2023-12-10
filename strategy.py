# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
# asset_index = 1  # only consider BTC (the **second** crypto currency in dataset)

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function ALWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use
import torch
import torch.nn
import pandas as pd
import numpy as np
from collections import deque, namedtuple
from train.model import *

model_file = ...  # Model's parameters
Volume_step = 5  # position volume step


def queue_append(memory_, data_):
    memory_.BTC.append(data_[0, :])
    memory_.ETH.append(data_[1, :])
    memory_.LTC.append(data_[2, :])
    memory_.XRP.append(data_[3, :])


def get_train_data(memory_):
    btc_m = torch.mean(torch.Tensor(np.vstack(memory_.BTC)), dim=0)
    eth_m = torch.mean(torch.Tensor(np.vstack(memory_.ETH)), dim=0)
    ltc_m = torch.mean(torch.Tensor(np.vstack(memory_.LTC)), dim=0)
    xrp_m = torch.mean(torch.Tensor(np.vstack(memory_.XRP)), dim=0)

    btc_x = (torch.Tensor(np.vstack(memory_.BTC)[-60:]) / btc_m).unsqueeze(0)
    eth_x = (torch.Tensor(np.vstack(memory_.ETH)[-60:]) / eth_m).unsqueeze(0)
    ltc_x = (torch.Tensor(np.vstack(memory_.LTC)[-60:]) / ltc_m).unsqueeze(0)
    xrp_x = (torch.Tensor(np.vstack(memory_.XRP)[-60:]) / xrp_m).unsqueeze(0)
    return btc_x, eth_x, ltc_x, xrp_x


def get_volume_mean(memory_, step=60):
    btc_v = np.array(list(memory_.BTC)[-step:])
    eth_v = np.array(list(memory_.ETH)[-step:])
    ltc_v = np.array(list(memory_.LTC)[-step:])
    xrp_v = np.array(list(memory_.XRP)[-step:])
    # mean_vol_ = np.array(np.mean(btc_v[:, 4]), np.mean(eth_v[:, 4]), np.mean(ltc_v[:, 4]), np.mean(xrp_v[:, 4]))
    # return mean_vol_
    return np.mean(btc_v[:, 4]), np.mean(eth_v[:, 4]), np.mean(ltc_v[:, 4]), np.mean(xrp_v[:, 4])


def get_signal(model, x_tuple):
    model.eval()
    y_l = []
    with torch.no_grad():
        for x in x_tuple:
            y = model(x).squeeze(0)  # (1, 2)
            y_l.append(y)
    return y_l


def get_signal_2(pred_prob):
    signal_y = int(torch.argmax(pred_prob, dim=-1))
    if signal_y == 0:
        return -1
    else:
        return 1


def get_opening_position(siganal_list, mean_vol):
    """
    Output the current opening position of 4 bitcoins
    :param siganal_list: [(0-prob, 1-prob), ...]
    :param mean_vol: array(vol0, vol1, vol2,...)
    :return: np.array(op_btc, op_eth, op_ltc, op_xrp)
    """
    mean_vol_ = np.array(mean_vol)
    position_ = np.repeat(0., 4)
    for idx, signal in enumerate(siganal_list):
        y = int(torch.argmax(signal, dim=-1))
        prob = float(signal[y])
        if y == 0:
            y = -1  # change signal

        if prob > 0.9:
            position_[idx] = 0.5 * y
        elif prob > 0.7:
            position_[idx] = 0.1 * y
        elif prob > 0.5:
            position_[idx] = 0.05 * y
        else:
            position_[idx] = 0.01 * y

    position_ = position_ * mean_vol_ # prevent the influence to market
    for i, p in enumerate(position_):
        position_[i] = round(p, 6)

    return position_


class Position:
    # Record the History Position changing
    def __init__(self, name, maxlen=5):
        self.name = name
        self.maxlen = maxlen

        self.time = deque(maxlen=maxlen)
        self.signal = deque(maxlen=maxlen)
        self.position = deque(maxlen=maxlen)

        self.total_position = 0.0

    def __len__(self):
        return len(self.time)

    def append(self, current_tsp):
        time_, signal_, position_ = current_tsp  # Current Position changing info
        self.time.append(time_)
        self.signal.append(signal_)
        self.position.append(position_)

    def update(self, current_tsp, current_position):
        self.total_position = current_position

        # Open new position based on current signal
        _, signal_, position_ = current_tsp  # Current Position changing info
        self.total_position = self.total_position + position_

        # Check length
        if len(self.time) != self.maxlen:
            self.append(current_tsp)
            return self.total_position

        # Close position 5 min ago
        self.total_position = self.total_position - self.position[0]

        # Check for extreme situation
        if signal_ == 0:
            if self.signal[-1] == 0 & self.signal[-2] == 0:  # Continuously Downside prediction
                idx = [index for index, value in enumerate(self.signal) if value == 1]  # Previously Upside position
                for i in idx:
                    self.total_position = self.total_position - self.position[i]  # Closing history Upward position
                    self.position[i] = 0.0
        else:
            if self.signal[-1] == 1 & self.signal[-2] == 1:
                idx = [index for index, value in enumerate(self.signal) if value == 0]
                for i in idx:
                    self.total_position = self.total_position - self.position[i]
                    self.position[i] = 0.0

        self.append(current_tsp)

        return self.total_position


def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant  交易费率
               cash_balance,  # your cash balance at current minute  现金价值
               crypto_balance,  # your crpyto currency balance at current minute  虚拟货币仓位价值
               total_balance,  # your total balance at current minute  总价值
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    if counter == 0:  # Initialization
        # Record Passed 60min data
        memory.BTC = deque(maxlen=1440)  # (60, 5:OHLCV)
        memory.ETH = deque(maxlen=1440)
        memory.LTC = deque(maxlen=1440)
        memory.XRP = deque(maxlen=1440)

        # Record the open position for each bit
        memory.BTC_position = Position('BTC')
        memory.ETH_position = Position('ETH')
        memory.LTC_position = Position('LTC')
        memory.XRP_position = Position('XRP')

        queue_append(memory, data)

        position = position_current
        memory = memory
        return position, memory

    else:
        queue_append(memory, data)

        if len(memory.BTC) == 1440:
            model = CNNALSTM()
            # model.load_state_dict(torch.load(model_file))  # Loading Trained Parameters

            btc_x, eth_x, ltc_x, xrp_x = get_train_data(memory)

            signal_list = get_signal(model, (btc_x, eth_x, ltc_x, xrp_x))  # [(0-prob, 1-prob), ...]
            volume_list = get_volume_mean(memory, step=Volume_step)  # array(vol0, vol1, vol2,...)
            open_position = get_opening_position(signal_list, volume_list)

            position = np.repeat(0., 4)
            for id, op in enumerate(open_position):
                signal_ = get_signal_2(signal_list[id])
                current_tsp_ = (time, signal_, open_position[id])

                if id == 0:
                    position[id] = memory.BTC_position.update(current_tsp_, position_current[id])
                elif id == 1:
                    position[id] = memory.ETH_position.update(current_tsp_, position_current[id])
                elif id == 2:
                    position[id] = memory.LTC_position.update(current_tsp_, position_current[id])
                elif id == 3:
                    position[id] = memory.XRP_position.update(current_tsp_, position_current[id])
                else:
                    raise Exception('Wrong idx num!')

            return position, memory
        else:  # Do Nothing
            position = position_current
            memory = memory
            return position, memory
