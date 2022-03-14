import numpy as np
import random
import math

class skl_proficiency():
    def __init__(self,vec_plm,vec_skl,q_matrix,arr_train,path,a):
        self.vec_plm = vec_plm
        self.vec_skl = vec_skl
        self.q_matrix = q_matrix
        self.arr_train = arr_train
        self.path = path
        self.exm_skl_sum = np.zeros([self.arr_train.shape[0],self.vec_skl.shape[0]])
        self.exm_skl_count = np.zeros([self.arr_train.shape[0], self.vec_skl.shape[0]])
        # print(self.exm_skl.shape)

        # TODO: the trait of all examinees;array;model parameters
        # random_theta
        # self.theta = np.random.random((self.arr_train.shape[0], 1))


        # normal
        self.arr_train_temp = self.arr_train[np.logical_not(np.isnan(self.arr_train))]
        self.arr_train_temp = np.reshape(self.arr_train_temp,(self.arr_train.shape[0],int(self.arr_train_temp.shape[0]/self.arr_train.shape[0])))
        self.mu = np.mean(self.arr_train_temp)
        self.sigma =np.var(self.arr_train_temp)
        # print('mu:',self.mu)
        # print('sigma:',self.sigma)
        self.sampleNo = self.arr_train.shape[0]
        self.theta = np.random.normal(self.mu, self.sigma, self.sampleNo)
        print('max_theta:',np.max(self.theta))
        print('min_theta:',np.min(self.theta))
        # self.theta = np.reshape(self.theta, (self.arr_train.shape[0], 1))
        self.theta = np.array(self.theta)[np.argsort(self.theta)]

        # sort
        self.mean = np.mean(self.arr_train_temp, axis=1)
        self.sort = np.argsort(self.mean)
        self.mean = np.array(self.mean)[np.argsort(self.mean)]
        self.mean = np.array(self.mean)[self.sort]
        self.mean = np.array(self.theta)[np.argsort(self.mean)]
        self.mean = np.reshape(self.mean, (self.arr_train.shape[0], 1))
        self.theta =self.mean

        # random_a
        # test a --1.7: 0.4721;2:0.4725---temp
        self.a = a
        # print('hyperp2-aIrtSlop:',self.a)


    def get_proficiency(self):
        for self.exm_count in range(self.arr_train.shape[0]):
            for self.plm_count in range(self.arr_train.shape[1]):
                # print(self.arr_train[self.exm_count][self.plm_count])
                # nan-pass

                # 1obj and 2sub: get the full score-master all the skills that the problem requires
                if self.arr_train[self.exm_count][self.plm_count] == 1:
                    for self.skl_count in range(self.q_matrix.shape[1]):
                        if self.q_matrix[self.plm_count][self.skl_count] == 1:
                            # print(self.skl_count)
                            self.exm_skl_sum[self.exm_count][self.skl_count] += 1
                            self.exm_skl_count[self.exm_count][self.skl_count] += 1

                # 3sub: get the zero score-do not master any skills that the problem requires
                if self.vec_plm[self.plm_count][0] == 1 and self.arr_train[self.exm_count][self.plm_count] == 0:
                    for self.skl_count in range(self.q_matrix.shape[1]):
                        if self.q_matrix[self.plm_count][self.skl_count] == 1:
                            # print(self.skl_count)
                            self.exm_skl_count[self.exm_count][self.skl_count] += 1
                # print(self.exm_skl)
                # print(self.exm_skl_count)

                # 4 obj: get the zero score-master a part of skills that the problem requires
                if self.vec_plm[self.plm_count][0] == 0 and self.arr_train[self.exm_count][self.plm_count] == 0:
                    for self.skl_count in range(self.q_matrix.shape[1]):
                        if self.q_matrix[self.plm_count][self.skl_count] == 1:
                            # consider
                            self.d_i_k = 1 / (1 + math.exp((-1) * self.a * (self.vec_skl[self.skl_count][0] - self.theta[self.exm_count][0])))
                            self.exm_skl_sum[self.exm_count][self.skl_count] += self.d_i_k
                            self.exm_skl_count[self.exm_count][self.skl_count] += 1
                            # do not consider
                            # self.exm_skl_count[self.exm_count][self.skl_count] += 1

                # 5 sub: get a part of scores-master a part of skills that the problem requires
                if self.vec_plm[self.plm_count][0] == 1 and 0 < self.arr_train[self.exm_count][self.plm_count] < 1:
                    for self.skl_count in range(self.q_matrix.shape[1]):
                        if self.q_matrix[self.plm_count][self.skl_count] == 1:
                            # consider
                            self.d_i_k = 1 / (1 + math.exp((-1) * self.a * ( self.theta[self.exm_count][0]-self.vec_skl[self.skl_count][0])))
                            self.exm_skl_sum[self.exm_count][self.skl_count] += self.d_i_k
                            self.exm_skl_count[self.exm_count][self.skl_count] += 1
                            # do not consider
                            # self.exm_skl_sum[self.exm_count][self.skl_count] += 1
                            # self.exm_skl_count[self.exm_count][self.skl_count] += 1

        # print('exm_skl_sum',self.exm_skl_sum)
        # print('exm_skl_count', self.exm_skl_count)
        self.exm_skl = self.exm_skl_sum/self.exm_skl_count
        # print('exm_skl',self.exm_skl)
        for self.exm_count in range(self.arr_train.shape[0]):
            for self.skl_count in range(self.q_matrix.shape[1]):
                # if skl is not obtained, the value is the average of other examinees' skills
                if np.isnan(self.exm_skl[self.exm_count][self.skl_count].item()):
                    self.exm_skl[self.exm_count][self.skl_count] = self.exm_skl_count.sum(axis=0)[self.skl_count]/(self.arr_train.shape[0]-np.isnan(self.exm_skl[:,self.skl_count]).sum())

        return self.exm_skl
        # print('exm_skl', self.exm_skl)
        # print(self.exm_skl.shape)


