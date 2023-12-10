import h5py
import pandas as pd
import numpy as np
import copy
import os
import sys
import matplotlib

# Change the working directory to your strategy folder.
# You should change this directory below on your own computer accordingly.
working_folder = os.getcwd()

# Write down your file paths for format 1 and format 2
# Note: You can test your strategy on different periods. Try to make your strategy profitable stably.
# e.g. format1_dir = '~/data/backtest_data_format1_week_final.h5'
#      format2_dir = '~/data/backtest_data_format2_week_final.h5'
format1_dir = r'./data/backtest_data_format1_week_final.h5'
format2_dir = r'./data/backtest_data_format2_week_final.h5'

# The following code is for backtestingc. DO NOT change it unless you want further exploration beyond the course project.
# import your handle_bar function
sys.path.append(working_folder)

# Run the main function in your demo.py to get your model and initial setup ready (if there is any)
os.chdir(working_folder)

# os.system('strategy.py')

# from Toymodel import handle_bar
from strategy import handle_bar


# Class of memory for data storage
class memory:
    def __init__(self):
        pass


class backTest:
    def __init__(self):
        # Initialize strategy memory with None. New memory will be updated every minute in backtest
        self.memory = memory()  # 初始化memory成空，但是后面可以返回新的memory

        # Initial setting of backtest
        self.init_cash = 100000.
        self.cash_balance_lower_limit = 10000.
        self.commissionRatio = 0.0005

        # Data path
        self.data_format1_path = format1_dir
        self.data_format2_path = format2_dir

        # You can adjust the path variables below to train and test your own model
        self.train_data_path = ''
        self.test_data_path = ''

    def pnl_analyze(self, strategyDetail):
        balance = strategyDetail.total_balance
        balance_hourly = balance.resample("H").last()
        ret_hourly = balance_hourly.pct_change()
        ret_hourly[0] = balance_hourly[0] / self.init_cash - 1
        ret_hourly.fillna(0, inplace=True)

        balance_daily = balance.resample("D").last()
        ret_daily = balance_daily.pct_change()
        ret_daily[0] = balance_daily[0] / self.init_cash - 1
        ret_daily.fillna(0, inplace=True)

        total_ret = balance[-1] / balance[0] - 1
        daily_ret = ret_daily.mean()
        sharpe_ratio = np.sqrt(365) * ret_daily.mean() / ret_daily.std()
        max_drawdown = (balance / balance.cummax() - 1).min()

        print("Total Return: ", total_ret)
        print("Average Daily Return: ", daily_ret)
        print("Sharpe Ratio: ", sharpe_ratio)
        print("Maximum Drawdown: ", max_drawdown)

        balance_hourly.plot(figsize=(12, 3), title='Balance Curve', grid=True)
        matplotlib.pyplot.show(block=True)

        pass

    def backTest(self):

        ''' Function that used to do back-testing based on the strategy you give
        Params: None
        
        Notes: this back-test function will move on minute bar and generate your 
        strategy detail dataframe by using the position vectors your strategy gives
        each minute
        '''

        format1 = h5py.File(self.data_format1_path, mode='r')
        format2 = h5py.File(self.data_format2_path, mode='r')
        assets = list(format1.keys())  # 不同的币种标签 BTC-ETH-LTC-XRP
        keys = list(format2.keys())  # 时间戳标签  2023-04-19-00-00-01

        for i in range(len(keys)):  # 对每个数据一一遍历
            data_cur_min = format2[keys[i]][:]  # 取出当前时间戳的四个币种，
            # 1. initialization
            if i == 0:
                total_balance = self.init_cash
                average_price_old = np.mean(data_cur_min[:, :4], axis=1)  # 四种货币的平均价格，当作初始成本价格
                position_old = np.repeat(0., 4)  # 四种货币的初始仓位，全是0
                position_new = np.repeat(0., 4)
                details = list()
                stop_signal = False

            # 2. calculate position & cash/crypto/total balance & transaction cost etc.
            position_change = position_new - position_old
            mask = np.abs(position_change) > .25 * data_cur_min[:, 4]  # 判断限制条件，是否超过0.25的成交量
            position_change[mask] = (.25 * data_cur_min[:, 4] * np.sign(position_change))[
                mask]  # 当超过0.25成交量时，会用0.25的给mask起来
            position_new = position_old + position_change
            average_price = np.mean(data_cur_min[:, :4], axis=1)  # 用四种货币的平均ohlc当作交易成本
            transaction_cost = np.sum(np.abs(position_change) * average_price * self.commissionRatio)  # 计算交易费用
            revenue = np.sum(position_old * (average_price - average_price_old)) - transaction_cost  # 价格变动导致原始仓位变化产生的收益
            crypto_balance = np.sum(np.abs(position_new * average_price))  # 仓位现在的总价值
            total_balance = total_balance + revenue  # 总价值 = 初始的总价值 + 仓位变动的收益  【更新仓位价值】 【自己算一下就明白公式了！】
            cash_balance = total_balance - crypto_balance  # 现金价值 = 总价值 - 现有的仓位价值
            # Detail: [po1, po2, po3, po4, cash_b, cryp_b, revenue, tot_b, trans_b]
            detail = np.append(position_new, [cash_balance, crypto_balance, revenue, total_balance, transaction_cost])
            details.append(copy.deepcopy(detail))

            position_old = copy.deepcopy(position_new)
            average_price_old = copy.deepcopy(average_price)

            # 3. check special cases
            # if cash balance is less than lower limit, the program will stop all trading actions in the future
            if (cash_balance < self.cash_balance_lower_limit) and (stop_signal == False):
                stop_signal = True
                print("Current cash balance is lower than", self.cash_balance_lower_limit)
                print("Your strategy is forced to stop")
                print("System will soon close all your positions (long and short) on crypto currencies")

            if stop_signal:
                position_new = np.repeat(0., 4)
                if '09:30:00' in keys[i]:
                    print(keys[i][:10])
                continue

            # Update position and memory
            [position_new, self.memory] = handle_bar(i,
                                                     keys[i],  # 时间
                                                     data_cur_min,  # 当前的数据
                                                     self.init_cash,  # 初始金额
                                                     self.commissionRatio,  # 交易费率
                                                     cash_balance,  # 现金
                                                     crypto_balance,  # 持仓
                                                     total_balance,  # 总价值
                                                     position_new,  # 当前仓位
                                                     self.memory)  # memory

            # Update position and timer
            if '09:30:00' in keys[i]:
                print(keys[i][:10])

        detailCol = assets + ["cash_balance", "crypto_balance", "revenue", "total_balance", "transaction_cost"]
        detailsDF = pd.DataFrame(details, index=pd.to_datetime(keys), columns=detailCol)

        format1.close()
        format2.close()
        return detailsDF


if __name__ == '__main__':
    ''' You can check the details of your strategy and do your own analyze by viewing 
    the strategyDetail dataframe
    '''
    bt = backTest()
    strategyDetail = bt.backTest()
    strategyDetail.to_csv(working_folder + "/backtest_details.csv")  # output backtest details to your working folder
    bt.pnl_analyze(strategyDetail)  # print performance summary, plot balance curve
