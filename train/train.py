"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/12/9 20:32
@File : train.py
"""
from init import *
from dataloader import *
from utils import *
from model import *


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, params, logger):
    logger.info(f'---------- Epoch {epoch} Training ---------')
    time1 = time.time()
    metric_monitor = MetricMonitor()
    model.train()

    for X, labels in train_loader:
        X = X.to(params['DEVICE'])
        labels = labels.to(params['DEVICE'])

        label_pre = model(X)
        loss = criterion(label_pre, labels)

        acc, f1_macro = get_accuracy_f1score(label_pre, labels)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('f1', f1_macro)
        metric_monitor.update('acc', acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info(
        f'Epoch {epoch}. Train {metric_monitor}. Time {time.strftime("%M:%S", time.gmtime(time.time() - time1))}s')
    scheduler.step()

    return (metric_monitor.metrics['Loss']['avg'],
            metric_monitor.metrics['acc']['avg'],
            metric_monitor.metrics['f1']['avg'])


def valid_one_epoch(epoch, model, valid_loader, criterion, params, logger):
    logger.info(f'---------- Epoch {epoch} Validating ---------')
    time1 = time.time()
    metric_monitor = MetricMonitor()
    model.eval()

    with torch.no_grad():
        for X, labels in valid_loader:
            X = X.to(params['DEVICE'])
            labels = labels.to(params['DEVICE'])

            label_pre = model(X)
            loss = criterion(label_pre, labels)

            acc, f1_macro = get_accuracy_f1score(label_pre, labels)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('f1', f1_macro)
            metric_monitor.update('acc', acc)

        logger.info(
            f'Epoch {epoch}. Valid: {metric_monitor}. Time {time.strftime("%M:%S", time.gmtime(time.time() - time1))}s')
    return (metric_monitor.metrics['Loss']['avg'],
            metric_monitor.metrics['acc']['avg'],
            metric_monitor.metrics['f1']['avg'])


def train_main(settings):
    logger = get_logger(f'{settings["MODEL"]["NAME"]}', settings['PATH']['LOGGER'])
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info(f'Using device: {device}')
    settings['DEVICE'] = device
    seed_setting(settings['MODEL']['SEED'])

    logger.info(f'====================== {settings["MODEL"]["NAME"]} Train ========================')

    # train_loader, valid_loader = get_train_valid_loader(settings)
    # train_loader, valid_loader = generate_dataloader_all(settings)
    train_loader, valid_loader = generate_dataloader_load(settings)

    model = CNNALSTM()
    model.to(settings['DEVICE'])
    model.apply(init_weight)

    criterion = nn.CrossEntropyLoss().to(settings['DEVICE'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['LR'], weight_decay=settings['WEIGHT_DECAY'])
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    early_stopping = [np.Inf]
    df_performance = pd.DataFrame()

    for epoch in range(settings['EPOCHS']):
        train_loss, train_acc, train_f1 = train_one_epoch(epoch, model, train_loader, criterion, optimizer,
                                                          scheduler, settings, logger)
        valid_loss, valid_acc, valid_f1 = valid_one_epoch(epoch, model, valid_loader, criterion, settings, logger)
        df_performance.loc[len(df_performance),
                           ['epoch',
                            'train_loss', 'train_acc', 'train_f1',
                            'valid_loss', 'valid_acc', 'valid_f1']] = (epoch,
                                                                       train_loss, train_acc, train_f1,
                                                                       valid_loss, valid_acc, valid_f1)

        if epoch > 2:
            if (valid_loss > early_stopping[-1]) & (
                    early_stopping[-1] > early_stopping[-2]):  # Early stopping setting

                model_save_path = os.path.join(settings['PATH']['MODEL'], f'model_{epoch}_{round(valid_acc, 3)}.pth')
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)

                logger.info(f'Early Stop: Best model is epoch_{epoch} model!')
                break
            elif epoch == settings['EPOCHS'] - 1:
                model_save_path = os.path.join(settings['PATH']['MODEL'], f'model_{epoch}_{round(valid_acc, 3)}.pth')
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Epochs End: Best model is epoch_{epoch} model!')

        early_stopping.append(valid_loss)

    df_performance.to_csv(f'train_performance.csv')
    del model, criterion, optimizer, scheduler
    del train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
