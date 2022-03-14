from Initial_data import DataSet
from QRCDM_model import QRCDM
from Test_Model import test_model
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch
import os

if __name__ == '__main__':
    obj_columns = ['obj_acc', 'obj_auc', 'obj_rmse', 'obj_mae']
    all_columns = ['obj_acc', 'obj_auc', 'obj_rmse', 'obj_mae', 'sub_rmse', 'sub_mae']
    result_table = pd.DataFrame(columns=['dataSet', 'counter', 'run_times'] + all_columns)  # 保存实验结果

    # ----------实验基本参数--------------
    basedir = 'D:/python_practice/认知诊断/QRCDM/'
    # basedir = '/home/y202699/python_project/'
    batch_size = 32
    lr = 9e-3
    epochs = 15
    device = 'cuda'

    dataSet_list = ('FrcSub', 'Math1', 'Math2', 'ASSIST_0910', 'ASSIST_2017')
    experiment_num = 10
    # ----------------------------

    result_index = 0
    for data_set_name in dataSet_list:
        # ----------数据集基本参数--------------
        dataSet = DataSet(basedir, data_set_name)
        Q = dataSet.get_Q()
        train_data = dataSet.train_data
        valid_data = dataSet.valid_data
        test_data = dataSet.test_data
        obj_prob_index = dataSet.obj_prob_index
        sub_prob_index = dataSet.sub_prob_index
        # ----------------------------

        total_stu_list = set(train_data.index) & \
            set(valid_data.index) & set(test_data.index)

        stu_idx_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                                    batch_size=batch_size, shuffle=True)
        model = QRCDM(Q=Q, lr=lr, device=device)

        for counter in range(experiment_num):

            model.train_model(stu_idx_loader, train_data, valid_data,
                              obj_prob_index, sub_prob_index, epochs=epochs)

            # model.save_parameter(dataSet.save_dir)

            W_ = model.W_.cpu().detach().numpy()
            D_ = model.D_.cpu().detach().numpy()
            guess_ = model.guess_.cpu().detach().numpy()
            miss_ = model.miss_.cpu().detach().numpy()

            test_result = test_model(W_, D_, guess_, miss_, stu_idx_loader, train_data, test_data,
                                     obj_prob_index, sub_prob_index, device='cpu')
            # 保存实验结果
            result_table.loc[result_index, 'dataSet'] = data_set_name
            result_table.loc[result_index, 'counter'] = counter
            if len(test_result) == 4:
                result_table.loc[result_index, obj_columns] = test_result
                result_index += 1
            if len(test_result) == 6:
                result_table.loc[result_index, all_columns] = test_result
                result_index += 1

            result_table.to_csv(os.path.join(basedir, 'output', 'result_table.csv'), index=False)
