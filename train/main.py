"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/12/9 20:32
@File : main.py
"""
import yaml

from init import *
from train import *


parser = argparse.ArgumentParser(description='CryptoTrading')
parser.add_argument('-s', '--settings', type=str, required=True, metavar='',
                    help='Location of your Training configs file')
args = parser.parse_args()


if __name__ == '__main__':
    settings = yaml.load(open(args.settings, 'r'), Loader=yaml.FullLoader)
    # settings = yaml.load(open(r'./config/ALSTM5.yaml', 'r'), Loader=yaml.FullLoader)
    train_main(settings)
