# difficulty+key words
# problem [obj or sub; full score; plm_difficulty;plm_keyw]
#skill  [ skl_difficulty;skl_keyw]

import numpy as np
import math
np.set_printoptions(threshold=np.inf)



class vector():
    def __init__(self,plm_intro,q_matrix,arr_train,path,kewy_num):
        self.path = path
        self.plm_intro = plm_intro
        self.arr_train = arr_train
        self.q_matrix = q_matrix
        self.keyw_num = kewy_num
        # print('hyperp1-keyw_num:', self.keyw_num)
        self.num_keyws =  self.get_keywslist() # Math1-188
        # print(self.num_keyws)
        if self.keyw_num <5:
            self.index = 5
        elif self.keyw_num<10:
            self.index = 6
        else:
            self.index = 7
        # print('num_keyws', self.num_keyws)
        self.arr_temp = np.nan * np.ones([self.plm_intro.shape[0], 2])


# TODO: difficulty
    def plm_difficulty(self):
        self.li_dft_all = []
        for self.p_count in range(self.arr_train.shape[1]):
            self.plm_tcount = 0
            self.plm_acount = 0
            for self.s_count in range(self.arr_train.shape[0]):
                if self.arr_train[self.s_count][self.p_count]>=self.plm_intro[self.p_count][1]/2:
                    self.plm_tcount+=1
                if self.arr_train[self.s_count][self.p_count]>=0:
                    self.plm_acount+=1
            # print('p', self.p_count)
            # print('plm_count',self.plm_acount)
            self.plm_d = (self.plm_acount-self.plm_tcount)/self.plm_acount   # false_num/all_num

            self.li_dft = self.get_difficulty_vec(self.plm_d)
            self.li_dft_all += self.li_dft

            self.arr_temp[self.p_count][0]=self.plm_tcount
            self.arr_temp[self.p_count][1] = self.plm_acount

        self.plm_dft = np.array(self.li_dft_all)
        self.plm_dft=np.reshape(self.plm_dft,(self.arr_train.shape[1],len(self.li_dft)))
        # print(self.plm_dft.shape)

        # [obj or sub; full score; plm_difficulty,keywords]
        # +difficulty
        self.plm_vec1 = np.append(self.plm_intro, self.plm_dft, axis=1)
        self.plm_vec = np.append(self.plm_vec1, values=np.zeros([self.plm_intro.shape[0], self.num_keyws]), axis=1)
        return self.plm_vec1.shape[1]
        # -difficulty
        # self.plm_vec = np.append(self.plm_intro, values=np.zeros([self.plm_intro.shape[0], self.num_keyws]), axis=1)
        # return self.plm_intro.shape[1]



    def skl_difficulty(self):
        self.li_dft_all = []
        for self.k_count in range(self.q_matrix.shape[1]):
            self.skl_tcount=0
            self.skl_acount=0
            for self.p_count in range(self.q_matrix.shape[0]):
                if self.q_matrix[self.p_count][self.k_count]==1:
                    self.skl_tcount += self.arr_temp[self.p_count][0]
                    self.skl_acount += self.arr_temp[self.p_count][1]
            self.skl_d = (self.skl_acount-self.skl_tcount)/self.skl_acount

            self.li_dft = self.get_difficulty_vec(self.skl_d)
            self.li_dft_all += self.li_dft

        self.skl_dft = np.array(self.li_dft_all)
        self.skl_dft= np.reshape(self.skl_dft,(self.q_matrix.shape[1], len(self.li_dft)))

        # [skl_difficulty, keywords]
        # +difficulty
        self.skl_vec =  np.append(self.skl_dft, values=np.zeros([self.q_matrix.shape[1], self.num_keyws]), axis=1)
        return len(self.li_dft)
        # -difficulty
        # self.skl_vec = np.zeros([self.q_matrix.shape[1], self.num_keyws])
        # return 0


    def get_difficulty_vec(self,dft):
        li_dft = []

        for i in range(1,self.index):
            self.interval = 1/math.pow(2,i)
            lvalue=0
            for j in range(int(math.pow(2,i))):
                if j==0 and dft ==0:
                    li_dft.append(1)
                elif  lvalue<dft<=lvalue+self.interval:
                    li_dft.append(1)
                else:
                    li_dft.append(0)
                lvalue+=self.interval
        return li_dft


    #TODO: keywords
    def get_keywslist(self):
        self.li_keyws = []
        for self.skl_name in open(self.path + 'qnames.txt'):
            self.skl_name = self.skl_name.strip('\n')
            if self.skl_name.lower() not in self.li_keyws:
                self.li_keyws.append(self.skl_name.lower())
            self.count_keyws = 0
            for self.keyw in open(self.path + 'skl_keyw/' + self.skl_name + '.txt', encoding='utf-8-sig'):
                # hyperparameter:keyw_num
                if self.count_keyws <self.keyw_num:
                    # change1: the number of wanted keywords-3
                    if self.keyw.strip('\n') not in self.li_keyws:
                        self.li_keyws.append(self.keyw.strip('\n'))
                        self.count_keyws += 1
                else:
                    break
        # print(self.li_keyws)
        return len(self.li_keyws)

    def skl_keyws(self,pre_skl):
        self.axis_0 = 0
        for self.skl_name in open(self.path + 'qnames.txt'):
            self.skl_name = self.skl_name.strip('\n')
            self.count_keyw = 0
            for self.keyw in open(self.path + 'skl_keyw/' + self.skl_name + '.txt', encoding='utf-8-sig'):
                if self.count_keyw <self.keyw_num:
                    # change2: the number of wanted keywords-3
                    self.loc = self.li_keyws.index(self.keyw.strip('\n'))
                    self.skl_vec[self.axis_0][self.loc+pre_skl] = 1
                    self.count_keyw += 1
                else:
                    break
            self.axis_0 += 1
        # print(self.skl_vec.shape,self.skl_vec)
        return self.skl_vec

    def plm_keyws(self,pre_plm):
        for self.plm_num in range(20):
            self.count_keyw = 0
            for self.keyw in open(self.path + 'plm_keyw/' + str(self.plm_num) + '.txt', encoding='utf-8-sig'):
                if self.count_keyw <self.keyw_num:
                    # change3: the number of wanted keywords-3
                    self.loc = self.li_keyws.index(self.keyw.strip('\n'))
                    self.plm_vec[self.plm_num][self.loc + pre_plm] = 1
                    self.count_keyw += 1
                else:
                    break
        # print(self.plm_vec.shape,self.plm_vec)
        return self.plm_vec,self.num_keyws

