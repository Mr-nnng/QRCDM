'''
Lina Gao (linafreee@163.com)
'''

import time
import warnings
from data_preprocessing import data_divide, key_words
from vector_representation import vector
from skill_proficiency import skl_proficiency
from deep_network import network_main
import numpy as np

warnings.filterwarnings("ignore")

# TODO: data processing


class pro_data():
    def get_data(self, ratio_num, fold):
        # divide data
        self.cl_dd = data_divide(path)
        # self.cl_dd.divide()  # generate data;one use
        self.arr_data, self.plm_intro, self.q_matrix, self.arr_train, self.arr_valid = self.cl_dd.load_data(
            ratio_num, fold)

        # get key words
        # self.cl_kw = key_words(path)
        # self.cl_kw.skl_keyw()   # get key words of skills from txt
        # self.cl_kw.blm_keyw(self.q_matrix)  # get key words of problems
        return self.arr_data, self.plm_intro, self.q_matrix, self.arr_train, self.arr_valid


# TODO: problem difficulty and skill difficulty
class vec_present():

    def get_vector(self):
        self.cl_v = vector(plm_intro, q_matrix, arr_train, path, hyper1)

        # difficulty
        self.pre_plm = self.cl_v.plm_difficulty()
        self.pre_skl = self.cl_v.skl_difficulty()

        # keywords
        # problem [obj 0 or sub 1; full score; plm_difficulty;plm_keyw]
        self.vec_skl = self.cl_v.skl_keyws(self.pre_skl)
        self.vec_plm, self.num_keyws = self.cl_v.plm_keyws(
            self.pre_plm)  # skill  [ skl_difficulty;skl_keyw]
        # print('self.vec_skl',self.vec_skl)

        return self.vec_skl, self.vec_plm, self.num_keyws


# TODO:skill proficiency
class skl_master():
    def get_skl_proficiency(self):
        self.cl_sp = skl_proficiency(
            vec_plm, vec_skl, q_matrix, arr_train, path, hyper2)
        self.skl_pfc = self.cl_sp.get_proficiency()
        return self.skl_pfc

# TODO: problem proficiency&grade prediction


class network_keras():
    def get_grades(self):
        # self.cl_el = network_main(vec_skl,vec_plm,num_keyws,q_matrix,skl_pfc,arr_train,arr_vali,hyper3,hyper4,hyper5,acitvation)  # train--vali
        self.cl_el = network_main(vec_skl, vec_plm, num_keyws, q_matrix, skl_pfc,
                                  arr_train, arr_valid, hyper3, hyper4, hyper5, acitvation)  # train_vali--test
        rmse, mae, acc_obj, auc_obj, rmse_obj, mae_obj, rmse_sub, mae_sub = self.cl_el.model()
        return rmse, mae, acc_obj, auc_obj, rmse_obj, mae_obj, rmse_sub, mae_sub


if __name__ == "__main__":
    print("start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    warnings.filterwarnings('ignore')
    # path = '../../Data/Math2/'  # Math1;Math2;FrcSub
    # path = 'D:/python_practice/认知诊断/自定义模型-认知诊断/deepCDF/Data/Math1/'
    path = 'D:/python_practice/认知诊断/SECDM/data/Math2015/FrcSub/'
    print(path)

    li_ra = [8]
    for ratio_num in li_ra:
        print('ratio_num:', ratio_num)
        '''
        Math1: ********************************************************************
        hyperp1-keyw_num==analysis-[1,2,3,4,5,6,7] [2]

        hyperp2-aIrtSlop== analysis-[0.5,1,1.2,1.5,1.7,2,2.5] [1.7]

        hyperp3-hidden_dim==analysis-[5, 10, 15, 20,25, 30,35] [20]

        hyperp4-batch_size==analysis-[1, 3, 1403,4209] [1]

        hyperp5-reg_l2==analysis-[0.005,0.01,0.03,0.05, 0.1,0.5,1,1.5][0.05]

        activation == 'tanh'


        Math2: **********************************************************************
        hyperp1-keyw_num== analysis-[1,3,5,7,9,11,13] [5]

        hyperp2-aIrtSlop== analysis-[0.5,1,1.2,1.5,1.7,2,2.5] [1.7]

        hyperp3-hidden_dim==analysis-[5,10, 15,20,25,30,35] [10]

         hyperp4-batch_size==analysis-[1,3911] [1]

         hyperp5-reg_l2==analysis-[0.01,0.03,0.05,0.1,0.5,1,1.5]  [0.01]

         activation ==fix-['tanh']
        '''

        # Math1
        li_hyper1_tune = [13]  # [3]
        li_hyper2_tune = [1.7]  # [1.7]
        li_hyper3_tune = [10]  # [20]
        li_hyper4_tune = [1]   # [1]
        li_hyper5_tune = [0.01]  # [ 0.03]
        # Math2
        # li_hyper1_tune = [5]  # [5]
        # li_hyper2_tune = [1.7]  # [1.7]
        # li_hyper3_tune = [10]   # [10]
        # li_hyper4_tune = [1]  # [1]
        # li_hyper5_tune =[0.01]  # [0.01]
        # ['softmax', 'tanh', 'relu', 'sigmoid', 'linear']
        li_activation = ['tanh']
        for hyper1 in li_hyper1_tune:
            for hyper2 in li_hyper2_tune:
                for hyper3 in li_hyper3_tune:
                    for hyper4 in li_hyper4_tune:
                        for hyper5 in li_hyper5_tune:
                            for acitvation in li_activation:
                                print('*' * 200)
                                print('hyperp1-keyw_num:', hyper1)
                                print('hyperp2-aIrtSlop:', hyper2)
                                print('hyperp3-hidden_dim:', hyper3)
                                print('hyperp4-batch_size:', hyper4)
                                print('hyperp5-reg_l2:', hyper5)
                                print('activation:', acitvation)
                                li_rmse = []
                                li_mae = []
                                # train valid--5fold [1,5](0,6); train_valid test--1time fold=0(0,1)
                                for i in range(1, 2):
                                    print(i, 'results')
                                    cl_pd = pro_data()
                                    arr_data, plm_intro, q_matrix, arr_train, arr_valid = cl_pd.get_data(
                                        ratio_num=ratio_num, fold=i)
                                    # print('exm_num:', arr_train.shape[0])
                                    #
                                    # difficulty# data
                                    cl_vd = vec_present()
                                    vec_skl, vec_plm, num_keyws = cl_vd.get_vector()
                                    cl_sm = skl_master()
                                    skl_pfc = cl_sm.get_skl_proficiency()
                                    cl_nk = network_keras()
                                    rmse, mae, acc_obj, auc_obj, rmse_obj, mae_obj, \
                                        rmse_sub, mae_sub = cl_nk.get_grades()
                                    print('=' * 100)
                                    print('rmse:', rmse)
                                    print('mae:', mae)
                                    print('obj:', acc_obj, auc_obj,
                                          rmse_obj, mae_obj)
                                    print('sub:', rmse_sub, mae_sub)
                                    print('=' * 100)
                                    li_rmse.append(rmse)
                                    li_mae.append(mae)
                                    # print('embed_dim:',vec_skl.shape[1])
                                # print('=' * 100)
                                # print('rmse:', np.mean(li_rmse))
                                # print('mae:', np.mean(li_mae))
                                # print('=' * 100)

    print("finish time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
