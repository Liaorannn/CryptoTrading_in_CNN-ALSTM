"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/12/8 2:06
@File : utils.py
"""
from init import *


# ============================= Data Preprocessing =================================
def map_label(x):
    return int(x > 0)


def data_processing(df_, name, train_d='2021-09-14', valid_d='2023-01-01', test_d='2023-04-19'):
    df_['time'] = pd.to_datetime(df_['time'])
    df_['Target5'] = df_['Close'].shift(-5) / df_['Close'] - 1
    df_['Target30'] = df_['Close'].shift(-30) / df_['Close'] - 1
    df_['TargetV'] = df_['Volume'][::-1].rolling(5).mean()[::-1]  # future 5 min mean Vol

    df_['Open'] = df_['Open'] / df_['Open'].rolling(1440).mean()
    df_['High'] = df_['High'] / df_['High'].rolling(1440).mean()
    df_['Low'] = df_['Low'] / df_['Low'].rolling(1440).mean()
    df_['Close'] = df_['Close'] / df_['Close'].rolling(1440).mean()
    df_['Volume'] = df_['Volume'] / df_['Volume'].rolling(1440).mean()

    df_ = df_.dropna().reset_index(drop=True)

    df_['Target5'] = df_['Target5'].apply(map_label)
    df_['Target30'] = df_['Target30'].apply(map_label)

    df_train = df_[(df_['time'] >= pd.to_datetime(train_d)) & (df_['time'] < pd.to_datetime(valid_d))].reset_index(
        drop=True)
    df_valid = df_[(df_['time'] >= pd.to_datetime(valid_d)) & (df_['time'] < pd.to_datetime(test_d))].reset_index(
        drop=True)
    # df_train = df_[df_['time'].between(pd.to_datetime(train_d), pd.to_datetime(valid_d))].reset_index(drop=True)
    # df_valid = df_[df_['time'].between(pd.to_datetime(valid_d), pd.to_datetime(test_d))].reset_index(drop=True)

    df_train.to_pickle(fr'./data/{name}_train.pkl')
    df_valid.to_pickle(fr'./data/{name}_valid.pkl')
    return df_train, df_valid


# ============================== Metrics Function ================================
def get_accuracy_f1score(output, y_true):
    # y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(output, dim=1).cpu()
    y_true = y_true.cpu()
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='macro'))


class MetricMonitor:
    def __init__(self):
        self.metrics = None
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0,
                                            'count': 0,
                                            'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return " | ".join([f'{metric_name}: {round(metric["avg"], 3)}' for metric_name, metric in self.metrics.items()])


# ============================== Model Training ===========================
def seed_setting(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


# ========================== Logger =============================
def get_logger(log_name, file_name, infer=False):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    if infer:
        file_name = os.path.join(file_name, f'{log_name}_valid.log')
    else:
        file_name = os.path.join(file_name, f'{log_name}_train.log')

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file_handler = logging.FileHandler(filename=file_name, mode='w')

    standard_formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(filename)s line:%(lineno)d] %(levelname)s %(message)s')
    simple_formatter = logging.Formatter('%(levelname)s %(message)s')

    console_handler.setFormatter(simple_formatter)
    file_handler.setFormatter(standard_formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
