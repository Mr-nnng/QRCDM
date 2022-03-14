# divide datasets; get key words

import numpy as np
import random
from textrank4zh import TextRank4Keyword

# TODO: divide the datasets
class data_divide():
    def __init__(self,path):
        self.divide_ratio = [0.2,0,0.8]  # [0.5,0.2,0.3];expand-[0.2,0,0.8]
        self.divide_num = [int(i*20) for i in self.divide_ratio]
        self.path = path
        self.arr_data = np.loadtxt(self.path+'data.txt')   # txt->array
        self.plm_intro = np.loadtxt(self.path+'problemdesc.txt')  # type and full score of problems;each datasets has 20problems
        self.q_matrix = np.loadtxt(self.path+'q.txt')   #  problems required the skills or not

    def divide(self):
        #  initialize
        self.arr_train =np.nan * np.ones(np.shape(self.arr_data))  # the shape of arr_train is same as that of arr_data
        self.arr_vali = np.nan * np.ones(np.shape(self.arr_data))  # the shape of arr_vali is same as that of arr_data
        self.arr_test = np.nan * np.ones(np.shape(self.arr_data))  # the shape of arr_test is same as that of arr_data
        self.arr_train_vali = np.nan * np.ones(np.shape(self.arr_data))
        # insure each set  has at least one student's data
        self.count = np.isnan(self.arr_train).sum()
        print("all_data count:",self.count)
        for self.exm_count in range(self.arr_data.shape[0]):
            # random
            self.li_ran = [0,1, 2, 3, 4, 5, 6,7, 8, 9,10,11, 12, 13,14,15, 16, 17, 18, 19]  # each dataset has 20 problems
            self.slice1 = random.sample(self.li_ran, self.divide_num[0])
            self.li_ran2 = list(set(self.li_ran) - set(self.slice1))
            self.slice2 = random.sample(self.li_ran2, self.divide_num[1])
            self.slice2 = self.slice2
            self.slice3 = list(set(self.li_ran2) - set(self.slice2))
            self.slice4= self.slice1+self.slice2

            for self.slice1_count in self.slice1:
                self.arr_train[self.exm_count][self.slice1_count] = self.arr_data[self.exm_count][self.slice1_count]
            for self.slice2_count in self.slice2:
                self.arr_vali[self.exm_count][self.slice2_count] = self.arr_data[self.exm_count][self.slice2_count]
            for self.slice3_count in self.slice3:
                self.arr_test[self.exm_count][self.slice3_count] = self.arr_data[self.exm_count][self.slice3_count]
            for self.slice4_count in self.slice4:
                self.arr_train_vali[self.exm_count][self.slice4_count] = self.arr_data[self.exm_count][self.slice4_count]
        # print(self.arr_train,self.arr_vali,self.arr_test)
        # np.savetxt(self.path + 'divide_data/train_data.txt', self.arr_train)
        # np.savetxt(self.path + 'divide_data/vali_data.txt', self.arr_vali)
        np.savetxt(self.path + 'divide_data/test_data.txt', self.arr_test)
        np.savetxt(self.path + 'divide_data/train_vali_data.txt', self.arr_train_vali)


    def load_data(self,ratio_num,fold):
            # print(ratio_num,fold)
        # self.arr_train = np.loadtxt(self.path+ 'divide_data_5fold/test_ratio'+str(ratio_num)+'/train_data'+str(fold)+'.txt')
        # self.arr_valid = np.loadtxt(self.path + 'divide_data_5fold/test_ratio'+str(ratio_num)+'/valid_data'+str(fold)+'.txt')

        # print(ratio_num, fold)
        self.arr_train = np.loadtxt(self.path + 'divide_data_5fold/test_ratio'+str(ratio_num)+'/train_vali_data'+str(ratio_num)+'.txt')
        self.arr_valid = np.loadtxt(self.path + 'divide_data_5fold/test_ratio'+str(ratio_num)+'/test_data'+str(ratio_num)+'.txt')

        return self.arr_data,self.plm_intro,self.q_matrix,self.arr_train,self.arr_valid

    def divide_data_5fold(self):
        #3-14[3,3,3,3,2]; 2-16[4,3,3,3,3];4-12[3,3,2,2,2,2];6-8[2,2,2,1,1]; 8-4[1,1,1,1]
        self.arr_train_vali_5fold = np.loadtxt(self.path + 'divide_data_5fold/test_ratio/train_vali_data6.txt')
        self.count_all=self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_train_vali_5fold).sum()
        print(self.count_all,self.arr_train_vali_5fold.shape[0],self.count_all/self.arr_train_vali_5fold.shape[0])
        self.arr_train1,self.arr_train2,self.arr_train3,self.arr_train4,self.arr_train5 = np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold))
        self.arr_valid1, self.arr_valid2, self.arr_valid3, self.arr_valid4, self.arr_valid5= np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold)),np.nan * np.ones(np.shape(self.arr_train_vali_5fold))
        for i in range(self.arr_train_vali_5fold.shape[0]):
            self.count1_plm, self.count2_plm, self.count3_plm, self.count4_plm, self.count5_plm = 0, 0, 0, 0, 0
            self.choice_list_plm = [1, 2, 3, 4,5] #
            for j in range(self.arr_train_vali_5fold.shape[1]):
                if np.isnan(self.arr_train_vali_5fold[i][j]):
                    continue
                else:
                    self.ran=random.choices(self.choice_list_plm, k=1)[0]
                    if self.ran ==1:
                        self.arr_train1[i][j]=self.arr_train_vali_5fold[i][j]
                        self.arr_valid2[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid3[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid4[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid5[i][j] = self.arr_train_vali_5fold[i][j]
                        self.count1_plm+=1
                        if self.count1_plm ==2:
                            self.choice_list_plm.remove(1)
                    if self.ran ==2:
                        self.arr_train2[i][j]=self.arr_train_vali_5fold[i][j]
                        self.arr_valid1[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid3[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid4[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid5[i][j] = self.arr_train_vali_5fold[i][j]
                        self.count2_plm += 1
                        if self.count2_plm ==2:
                            self.choice_list_plm.remove(2)
                    if self.ran ==3:
                        self.arr_train3[i][j]=self.arr_train_vali_5fold[i][j]
                        self.arr_valid2[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid1[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid4[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid5[i][j] = self.arr_train_vali_5fold[i][j]
                        self.count3_plm += 1
                        if self.count3_plm ==2:
                            self.choice_list_plm.remove(3)
                    if self.ran ==4:
                        self.arr_train4[i][j]=self.arr_train_vali_5fold[i][j]
                        self.arr_valid2[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid3[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid1[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid5[i][j] = self.arr_train_vali_5fold[i][j]
                        self.count4_plm += 1
                        if self.count4_plm ==1:
                            self.choice_list_plm.remove(4)
                    if self.ran ==5:
                        self.arr_train5[i][j]=self.arr_train_vali_5fold[i][j]
                        self.arr_valid2[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid3[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid4[i][j] = self.arr_train_vali_5fold[i][j]
                        self.arr_valid1[i][j] = self.arr_train_vali_5fold[i][j]
                        self.count5_plm += 1
                        if self.count5_plm ==1:
                            self.choice_list_plm.remove(5)

        np.savetxt(self.path + 'divide_data_5fold/test_ratio/train_data1.txt', self.arr_train1)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/vali_data1.txt', self.arr_valid1)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/train_data2.txt', self.arr_train2)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/vali_data2.txt', self.arr_valid2)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/train_data3.txt', self.arr_train3)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/vali_data3.txt', self.arr_valid3)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/train_data4.txt', self.arr_train4)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/vali_data4.txt', self.arr_valid4)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/train_data5.txt', self.arr_train5)
        np.savetxt(self.path + 'divide_data_5fold/test_ratio/vali_data5.txt', self.arr_valid5)

        print(self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_train1).sum(),
              self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_valid1).sum(),
              self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_train2).sum(),
              self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_valid2).sum(),
              self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_train3).sum(),
              self.arr_train_vali_5fold.shape[0] * self.arr_train_vali_5fold.shape[1] -np.isnan(self.arr_valid3).sum(),
              self.arr_train_vali_5fold.shape[0] * self.arr_train_vali_5fold.shape[1] -np.isnan(self.arr_train4).sum(),
              self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_valid4).sum(),
              self.arr_train_vali_5fold.shape[0]*self.arr_train_vali_5fold.shape[1]-np.isnan(self.arr_train5).sum(),
              self.arr_train_vali_5fold.shape[0] * self.arr_train_vali_5fold.shape[1] -np.isnan(self.arr_valid5).sum())

# TODO: Key words
class key_words():
    def __init__(self,path):
        self.path=path

    def skl_keyw(self):
        # get skill names
        for self.skl_name in open(self.path+'qnames.txt'):
            self.skl_name=self.skl_name.strip('\n')
            print(self.skl_name)
            # get key words
            with open(self.path+'skl_doc/'+self.skl_name+'.txt', 'r') as self.skl_doc:
                # utf-8 code
                self.skl_txt = self.skl_doc.read()
            tr4w = TextRank4Keyword()
            tr4w.analyze(text=self.skl_txt, lower=True, window=2)
            print('key words：')
            for item in tr4w.get_keywords(30, word_min_len=1):
                print(item.word,item.weight)
                with open(self.path + 'skl_keyw/' + self.skl_name + '.txt', 'a+') as self.f:
                    self.f.write(item.word)
                    self.f.write('\n')
            print('----------------------------------------------------------------------------------------------------------------------------------------------------------')


    def blm_keyw(self,q_matrix):
        # get the skill names and save them as list
        self.li_skl = []
        for self.skl_name in open(self.path + 'qnames.txt'):
            self.skl_name = self.skl_name.strip('\n')
            self.li_skl.append(self.skl_name)
        print(self.li_skl)
        # represent the problems with skill names and their keywords
        self.q_matrix = q_matrix
        for self.p_count in range(self.q_matrix.shape[0]):
            print(self.p_count)
            self.keyw_count = 0  # assure the number of keywords of each problem is 20
            self.li_skl1 = []  # record the skills problem requires
            self.li_skl2 = []  # record all keywords of each problem
            # skill names
            for self.s_count in range(self.q_matrix.shape[1]):
                if self.q_matrix[self.p_count][self.s_count] == 1:
                    with open(self.path + 'plm_keyw/' + str(self.p_count) + '.txt', 'a+') as self.p_f:
                        self.p_f.write(self.li_skl[self.s_count].lower())
                        self.p_f.write('\n')
                    self.li_skl1.append(self.li_skl[self.s_count])
                    self.li_skl2.append(self.li_skl[self.s_count].lower())
                    print(self.li_skl[self.s_count].lower())
                    self.keyw_count += 1
            # keywords of the skills
            self.line_count = 0
            while self.keyw_count < 20:
                for self.skl_name in self.li_skl1:
                    with open(self.path + 'skl_keyw/' + self.skl_name + '.txt', 'r',encoding='utf-8-sig') as self.f:
                        self.line = self.f.readlines()
                        with open(self.path + 'plm_keyw/' + str(self.p_count) + '.txt', 'a+') as self.p_f:
                            if self.line[self.line_count].strip('\n') not in self.li_skl2:
                                self.li_skl2.append(self.line[self.line_count].split(',', 1)[0])
                                print(self.line[self.line_count].strip('\n'))
                                self.p_f.write(self.line[self.line_count].strip('\n'))
                                self.p_f.write('\n')
                                self.keyw_count +=1
                    if self.keyw_count ==20:
                        print('ok')
                        break
                self.line_count += 1


if __name__ == "__main__":
    path = '../../Data/Math2/'
    dd=data_divide(path)
    # dd.divide_data_5fold()







