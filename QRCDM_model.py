import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
from sklearn.model_selection import KFold


def cross_entropy_loss(pred_, labels):  # 交叉熵损失函数
    pred = pred_.clamp(1e-6, 1 - 1e-6)
    pred_log = torch.log(pred)
    one_minus_log = torch.log(1 - pred)
    loss = -1 * (labels * pred_log + (1 - labels) * one_minus_log)
    loss_mean = loss.mean()
    return loss_mean


def evaluate_obj(pred_, label):
    pred = np.array(pred_).round()

    acc = metrics.accuracy_score(label, pred)
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


def format_data(record, test_record, n_splits=5):
    train = [[], []]  # 习题，得分
    valid = [[], [], []]  # 学生,习题，得分
    test = [[], [], []]
    stu_list = set(record.index)

    KF = KFold(n_splits=n_splits, shuffle=True)  # 5折交叉验证
    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        if len(stu_item) >= n_splits:
            test_item = test_record.loc[[stu], 'item_id'].values - 1
            test_score = test_record.loc[[stu], 'score'].values
            for train_prob, valid_prob in KF.split(stu_item):
                train[0].append(stu_item[train_prob])
                train[1].append(stu_score[train_prob])

                valid[0].extend([count] * len(valid_prob))
                valid[1].extend(stu_item[valid_prob])
                valid[2].extend(stu_score[valid_prob])
                test[0].extend([count] * len(test_item))
                test[1].extend(test_item)
                test[2].extend(test_score)
                count += 1
    valid_data = []
    valid_data.append(torch.tensor(valid[0]).long())
    valid_data.append(torch.tensor(valid[1]).long())
    valid_data.append(torch.tensor(valid[2]).float())

    test_data = []
    test_data.append(torch.tensor(test[0]).long())
    test_data.append(torch.tensor(test[1]).long())
    test_data.append(torch.tensor(test[2]).float())

    return train, valid_data, test_data


class QRCDM():
    def __init__(self, Q, lr=1e-3, device='cpu'):
        self.device = torch.device(device)
        self.sigmoid = torch.nn.Sigmoid()
        self.skill_num = Q.shape[1]
        # --------------模型参数---------------------
        Q = Q.to(device)
        W_ = Q.clone()
        W_.requires_grad = True
        D_ = Q.clone()
        D_.requires_grad = True
        # 猜测率、失误率
        guess_ = torch.ones((1, Q.shape[0])).to(device) * -2
        guess_.requires_grad = True
        miss_ = torch.ones((1, Q.shape[0])).to(device) * -2
        miss_.requires_grad = True
        # ------------------------------------------
        self.W_ = W_
        self.D_ = D_
        self.guess_ = guess_
        self.miss_ = miss_
        self.optimizer = torch.optim.Adam([self.W_, self.D_, self.guess_, self.miss_], lr=lr)

    def forward(self, score_list, prob_list):  # 前向传播,传入得分列表和习题索引列表
        k = self.skill_num
        device = self.device
        # drop = self.drop
        W_ = self.W_
        D_ = self.D_
        guess_ = self.guess_
        miss_ = self.miss_
        sigmoid = self.sigmoid

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

    def train_model(self, stu_idx_loader, train_data, test_data,
                    obj_prob_index, sub_prob_index, epochs):
        device = self.device
        optimizer = self.optimizer

        for epoch in range(1, epochs + 1):
            valid_loss_list, test_loss_list = [], []
            # [[train_data],[valid_data]]
            obj_true_list, obj_pred_list = [[], []], [[], []]
            if sub_prob_index is not None:
                sub_true_list, sub_pred_list = [[], []], [[], []]

            for betch_data in tqdm(stu_idx_loader, "[Epoch:%s]" % epoch):
                stu_list = np.array([x.numpy()
                                    for x in betch_data], dtype='int').reshape(-1)
                train, valid, test = format_data(
                    train_data.loc[stu_list, :], test_data.loc[stu_list, :])

                # ----------训练集（起始）--------------
                pred = self.forward(train[1], train[0])
                valid_pred = pred[valid[0], valid[1]]
                valid_score = valid[2].float().to(device)
                loss = cross_entropy_loss(valid_pred, valid_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                valid_loss_list.append(loss.item())
                with torch.no_grad():
                    if sub_prob_index is not None:
                        valid_sub_index = [(x in list(sub_prob_index))for x in valid[1]]
                        valid_obj_index = [(x in list(obj_prob_index))for x in valid[1]]
                        if sum(valid_sub_index) > 0:
                            valid_sub_score = valid[2][valid_sub_index].tolist()
                            valid_sub_pred = valid_pred[valid_sub_index].clone().to('cpu').detach().tolist()
                            sub_true_list[0].extend(valid_sub_score)
                            sub_pred_list[0].extend(valid_sub_pred)
                        if sum(valid_obj_index) > 0:
                            valid_obj_score = valid[2][valid_obj_index].tolist()
                            valid_obj_pred = valid_pred[valid_obj_index].clone().to('cpu').detach().tolist()
                            obj_true_list[0].extend(valid_obj_score)
                            obj_pred_list[0].extend(valid_obj_pred)

                    else:
                        valid_obj_score = valid[2].tolist()
                        valid_obj_pred = valid_pred.clone().to('cpu').detach().tolist()
                        obj_true_list[0].extend(valid_obj_score)
                        obj_pred_list[0].extend(valid_obj_pred)
                # ----------训练集（终止）--------------

                # ----------验证集（起始）--------------
                with torch.no_grad():
                    test_pred = pred[test[0], test[1]].clone().to('cpu').detach()
                    test_loss = cross_entropy_loss(test_pred, test[2].float())
                    test_loss_list.append(test_loss.item())

                    if sub_prob_index is not None:
                        test_sub_index = [(x in list(sub_prob_index))for x in test[1]]
                        test_obj_index = [(x in list(obj_prob_index))for x in test[1]]
                        if sum(test_sub_index) > 0:
                            test_sub_score = test[2][test_sub_index].tolist()
                            test_sub_pred = test_pred[test_sub_index].clone().to('cpu').detach().tolist()
                            sub_true_list[1].extend(test_sub_score)
                            sub_pred_list[1].extend(test_sub_pred)
                        if sum(test_obj_index):
                            test_obj_score = test[2][test_obj_index].tolist()
                            test_obj_pred = test_pred[test_obj_index].clone().to('cpu').detach().tolist()
                            obj_true_list[1].extend(test_obj_score)
                            obj_pred_list[1].extend(test_obj_pred)
                    else:
                        test_obj_score = test[2].tolist()
                        test_obj_pred = test_pred.clone().to('cpu').detach().tolist()
                        obj_true_list[1].extend(test_obj_score)
                        obj_pred_list[1].extend(test_obj_pred)
                # ----------验证集（终止）--------------

            print("[TrainingEpoch: %d] loss:%.6f  valid_loss:%.6f" %
                  (epoch, np.mean(valid_loss_list), np.mean(test_loss_list)))
            if sub_prob_index is not None:
                obj_acc, obj_auc, obj_rmse, obj_mae = evaluate_obj(obj_pred_list[0], obj_true_list[0])
                sub_rmse, sub_mae = evaluate_sub(sub_pred_list[0], sub_true_list[0])
                print("\ttrain: \tobj_acc:%.6f, obj_auc:%.6f, obj_rmse:%.6f, obj_mae:%.6f, \n\t\tsub_rmse: % .6f, sub_mae: % .6f" % (
                    obj_acc, obj_auc, obj_rmse, obj_mae, sub_rmse, sub_mae))

                obj_acc, obj_auc, obj_rmse, obj_mae = evaluate_obj(obj_pred_list[1], obj_true_list[1])
                sub_rmse, sub_mae = evaluate_sub(sub_pred_list[1], sub_true_list[1])
                print("\tvalid: \tobj_acc:%.6f, obj_auc:%.6f, obj_rmse:%.6f, obj_mae:%.6f, \n\t\tsub_rmse: % .6f, sub_mae: % .6f" % (
                    obj_acc, obj_auc, obj_rmse, obj_mae, sub_rmse, sub_mae))
            else:
                obj_acc, obj_auc, obj_rmse, obj_mae = evaluate_obj(obj_pred_list[0], obj_true_list[0])
                print("\ttrain: \tobj_acc:%.6f, obj_auc:%.6f, obj_rmse:%.6f, obj_mae:%.6f" % (obj_acc, obj_auc, obj_rmse, obj_mae))
                obj_acc, obj_auc, obj_rmse, obj_mae = evaluate_obj(obj_pred_list[1], obj_true_list[1])
                print("\tvalid: \tobj_acc:%.6f, obj_auc:%.6f, obj_rmse:%.6f, obj_mae:%.6f" % (obj_acc, obj_auc, obj_rmse, obj_mae))

    def save_parameter(self, save_dir):
        # 存储参数F
        np.savetxt(save_dir + 'W_.txt', self.W_.cpu().detach().numpy())
        np.savetxt(save_dir + 'D_.txt', self.D_.cpu().detach().numpy())
        np.savetxt(save_dir + 'miss_.txt', self.miss_.cpu().detach().numpy())
        np.savetxt(save_dir + 'guess_.txt', self.guess_.cpu().detach().numpy())
        print('模型参数已成功保存！')
