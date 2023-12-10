"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/12/8 2:06
@File : init.py
"""
# 系统
import numpy as np
import pandas as pd
import os
import re
import gc
import random
import time
import yaml
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict, namedtuple

# 图片处理
import matplotlib.pyplot as plt

# 模型
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support