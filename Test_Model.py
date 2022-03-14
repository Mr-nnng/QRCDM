import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm


def evaluate_obj(pred, label):
    acc = metrics.accuracy_score(label, np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(label, pred)
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


def evaluate_sub(pred, label):
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return rmse, mae


def format_test(record, test_record):
    train = [[], []]  # 习题，得分
    test = [[], [], []]  # 学生,习题，得分
    stu_list = set(record.index)
    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        test_item = test_record.loc[[stu], 'item_id'].values - 1
        test_score = test_record.loc[[stu], 'score'].values
        train[0].append(stu_item)
        train[1].append(stu_score)
        test[0].extend([count] * len(test_item))
        test[1].extend(test_item)
        test[2].extend(test_score)
        count += 1

    test_data = []
    test_data.append(torch.tensor(test[0]).long())
    test_data.append(torch.tensor(test[1]).long())
    test_data.append(torch.tensor(test[2]).float())

    return train, test_data


def test_forward(W_, D_, guess_, miss_, score_list, prob_list, device):  # 前向传播
    device = torch.device(device)
    sigmoid = torch.nn.Sigmoid()
    W_ = torch.tensor(W_).float().to(device)
    D_ = torch.tensor(D_).float().to(device)
    guess_ = torch.tensor(guess_).float().to(device)
    miss_ = torch.tensor(miss_).float().to(device)

    k = W_.shape[1]
    A = torch.zeros(len(score_list), k).to(device)
    for i, X_i in enumerate(score_list):
        X_i = torch.tensor(X_i).float().to(device).reshape(1, -1)
        W_i = torch.softmax(W_[prob_list[i]], dim=0)
        A[i] = X_i @ W_i
    D = torch.softmax(D_, dim=1)
    Y_ = A @ D.T
    miss = sigmoid(miss_)
    guess = sigmoid(guess_)
    Y = (1 - miss) * Y_ + guess * (1 - Y_)
    return Y


def test_model(W_, D_, guess_, miss_, stu_idx_loader, train_data, test_data,
               obj_prob_index, sub_prob_index, device):
    obj_true_list, obj_pred_list = [], []
    if sub_prob_index is not None:
        sub_true_list, sub_pred_list = [], []
    for betch_data in tqdm(stu_idx_loader, 'Testing:'):
        stu_list = np.array([x.numpy()
                            for x in betch_data], dtype='int').reshape(-1)
        train, test = format_test(train_data.loc[stu_list, :], test_data.loc[stu_list, :])

        pred = test_forward(W_, D_, guess_, miss_, train[1], train[0], device)
        test_pred = pred[test[0], test[1]].clone().to('cpu').detach()

        if sub_prob_index is not None:
            test_sub_index = [(x in list(sub_prob_index))for x in test[1]]
            test_obj_index = [(x in list(obj_prob_index))for x in test[1]]
            if sum(test_sub_index) > 0:
                test_sub_score = test[2][test_sub_index].tolist()
                test_sub_pred = test_pred[test_sub_index].tolist()
                sub_true_list.extend(test_sub_score)
                sub_pred_list.extend(test_sub_pred)
            if sum(test_obj_index) > 0:
                test_obj_score = test[2][test_obj_index].tolist()
                test_obj_pred = test_pred[test_obj_index].tolist()
                obj_true_list.extend(test_obj_score)
                obj_pred_list.extend(test_obj_pred)
        else:
            test_obj_score = test[2].tolist()
            test_obj_pred = test_pred.clone().to('cpu').detach().tolist()
            obj_true_list.extend(test_obj_score)
            obj_pred_list.extend(test_obj_pred)

    if sub_prob_index is not None:
        obj_acc, obj_auc, obj_rmse, obj_mae = evaluate_obj(obj_pred_list, obj_true_list)
        sub_rmse, sub_mae = evaluate_sub(sub_pred_list, sub_true_list)
        print("\ttest: \tobj_acc:%.6f, obj_auc:%.6f, obj_rmse:%.6f, obj_mae:%.6f, \n\t\tsub_rmse: % .6f, sub_mae: % .6f" % (
            obj_acc, obj_auc, obj_rmse, obj_mae, sub_rmse, sub_mae))
        return obj_acc, obj_auc, obj_rmse, obj_mae, sub_rmse, sub_mae
    else:
        obj_acc, obj_auc, obj_rmse, obj_mae = evaluate_obj(obj_pred_list, obj_true_list)
        print("\ttest: \tobj_acc:%.6f, obj_auc:%.6f, obj_rmse:%.6f, obj_mae:%.6f" % (obj_acc, obj_auc, obj_rmse, obj_mae))
        return obj_acc, obj_auc, obj_rmse, obj_mae
