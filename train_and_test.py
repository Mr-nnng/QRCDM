from Initial_data import DataSet
from QRCDM_model import QRCDM
from Test_Model import test_model
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    # ----------基本参数--------------
    basedir = 'D:/python_practice/认知诊断/QRCDM/'
    dataSet_list = ('FrcSub', 'Math1', 'Math2', 'ASSIST_0910', 'ASSIST_2017')
    data_set_name = dataSet_list[1]
    batch_size = 32
    lr = 9e-3
    epochs = 15
    device = 'cuda'
    # ----------基本参数--------------

    dataSet = DataSet(basedir, data_set_name)
    Q = dataSet.get_Q()
    train_data = dataSet.train_data
    valid_data = dataSet.valid_data
    test_data = dataSet.test_data
    obj_prob_index = dataSet.obj_prob_index
    sub_prob_index = dataSet.sub_prob_index

    total_stu_list = set(train_data.index) & \
        set(valid_data.index) & set(test_data.index)
    train_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                              batch_size=batch_size, shuffle=True)

    model = QRCDM(Q=Q, lr=lr, device=device)
    model.train_model(train_loader, train_data, valid_data,
                      obj_prob_index, sub_prob_index, epochs=epochs)

    save_dir = dataSet.save_dir
    # model.save_parameter(dataSet.save_dir)

    print('数据集：', data_set_name)
    W_ = model.W_.cpu().detach().numpy()
    D_ = model.D_.cpu().detach().numpy()
    guess_ = model.guess_.cpu().detach().numpy()
    miss_ = model.miss_.cpu().detach().numpy()

    test_result = test_model(W_, D_, guess_, miss_, train_loader, train_data, test_data,
                             obj_prob_index, sub_prob_index, device='cpu')
