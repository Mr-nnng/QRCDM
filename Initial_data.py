import pandas as pd
import numpy as np
import torch


class DataSet():
    def __init__(self, basedir, dataSetName):
        self.basedir = basedir
        self.dataSetName = dataSetName
        if dataSetName == 'FrcSub':
            read_dir = basedir + '/data/frcSub/'
            save_dir = basedir + '/output/frcSub/'
            n = 536
            m = 20
            k = 8
        elif dataSetName == 'Math1':
            read_dir = basedir + '/data/math1/'
            save_dir = basedir + '/output/math1/'
            n = 4209
            m = 20
            k = 11
        elif dataSetName == 'Math2':
            read_dir = basedir + '/data/math2/'
            save_dir = basedir + '/output/math2/'
            n = 3911
            m = 20
            k = 16
        elif dataSetName == 'ASSIST_0910':
            read_dir = basedir + '/data/a0910/'
            save_dir = basedir + '/output/a0910/'
            n = 4163
            m = 17746
            k = 123
        elif dataSetName == 'ASSIST_2017':
            read_dir = basedir + '/data/a2017/'
            save_dir = basedir + '/output/a2017/'
            n = 1678
            m = 2210
            k = 101
        else:
            print('Dataset does not exist!')
            exit(0)
        print('数据集：', dataSetName)
        item = pd.read_csv(read_dir + "item.csv")

        train_data = pd.read_csv(read_dir + "train.csv").set_index('user_id')
        test_data = pd.read_csv(read_dir + "test.csv").set_index('user_id')

        if dataSetName in ('FrcSub', 'ASSIST_0910', 'ASSIST_2017'):
            if dataSetName != 'FrcSub':
                valid_data = pd.read_csv(
                    read_dir + "valid.csv").set_index('user_id')
            else:
                valid_data = pd.read_csv(
                    read_dir + "test.csv").set_index('user_id')
            obj_prob_index = 'All'
            sub_prob_index = None
        else:
            valid_data = pd.read_csv(read_dir + "test.csv").set_index('user_id')
            # type of problems
            obj_prob_index = np.loadtxt(
                read_dir + "obj_prob_index.csv", delimiter=',', dtype=int)
            sub_prob_index = np.loadtxt(
                read_dir + "sub_prob_index.csv", delimiter=',', dtype=int)

        self.total_stu_list = set(train_data.index) & \
            set(valid_data.index) & set(test_data.index)

        self.stu_num = n
        self.prob_num = m
        self.skill_num = k
        self.item = item
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.save_dir = save_dir
        self.obj_prob_index = obj_prob_index
        self.sub_prob_index = sub_prob_index

    def get_Q(self):
        Q = np.zeros((self.prob_num, self.skill_num), dtype='bool')
        item = self.item
        for idx in item.index:
            item_id = item.loc[idx, 'item_id']
            know_list = item.loc[idx, 'knowledge_code'].replace(
                '[', '').replace(']', '').split(',')
            for know in know_list:
                Q[item_id - 1, int(know) - 1] = True
        return torch.tensor(Q, dtype=torch.float)
