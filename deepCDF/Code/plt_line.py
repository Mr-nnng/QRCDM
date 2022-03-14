import matplotlib.pyplot as plt
import time
import numpy as np


plt.rc('font',family='Times New Roman')

class plt_parameter_analysis():
    def __init__(self):
        self.path = 'E:/dearling/Papers/1 Learning ability/图/'
        self.font_x = {'family': 'Times New Roman',
                        'color': '#000000',
                        'weight': 'normal',
                       'size': 12,  # 'size': 10,  # 18 for ratio
                        }
        self.font_y1 = {'family': 'Times New Roman',
                   'color': 'r',
                   'weight': 'normal',
                     'size': 14,  # 'size': 10,
                   }
        self.font_y2 = {'family': 'Times New Roman',
                   'color': 'b',
                   'weight': 'normal',
                    'size': 14,  # 'size': 10,
                   }

        self.font_y3 = {'family': 'Times New Roman',
                        'color': '#000000',
                        'weight': 'normal',

                        'size': 12,  # for test ratio  # 'size': 11,  18 for ratio
                        }


    def keyw_num(self):
        self.Math1 = [1,2,3,4,5,6,7]
        self.Math1_RMSE = [0.4325,0.409,0.4164,0.4225,0.434,0.4136,0.4324]
        self.Math1_MAE = [0.3714,0.3438,0.3604,0.3658,0.3778,0.3609,0.3751]

        self.Math2  = [1,3,5,7,9,11,13]
        self.Math2_RMSE = [0.4579,0.4676,0.442,0.4652,0.4601,0.4516,0.448]
        self.Math2_MAE = [0.398,0.4055,0.3721,0.4051,0.3906,0.3851,0.3847]  # 创建图并命名
        fig = plt.figure(figsize=(9, 3), facecolor='w')
        ax = fig.add_subplot(121)
        lns1 = ax.plot(self.Math1, self.Math1_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.40, 0.440)
        plt.yticks([0.40, (0.40+(0.40+0.440) / 2)/2,(0.40+0.440) / 2,((0.40+0.440) / 2+0.440)/2, 0.440])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math1, self.Math1_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.34, 0.38)
        plt.yticks([0.34, (0.34+(0.34+0.38) / 2)/2,(0.34+0.38) / 2,  ((0.34+0.38) / 2+0.38)/2,0.38])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs,loc='lower')
        ax.tick_params(direction='in')
        plt.xticks([1,2,3,4,5,6,7])
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$k_n$', self.font_x,)
        plt.xlim(np.min(self.Math1), np.max(self.Math1))
        plt.title('(a) Math1', fontsize=11, family='Times New Roman')

        ax = fig.add_subplot(122)
        lns1 = ax.plot(self.Math2, self.Math2_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.43, 0.470)
        plt.yticks([0.43, (0.43+(0.43+0.470) / 2)/2,(0.43+0.470) / 2,((0.43+0.470) / 2+0.470)/2, 0.470])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math2, self.Math2_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.36, 0.41)
        plt.yticks([0.36, (0.36+(0.36+0.41) / 2)/2,(0.36+0.41) / 2,  ((0.36+0.41) / 2+0.41)/2,0.41])
        plt.xticks([1,3,5,7,9,11,13])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs,loc='lower right')
        ax.tick_params(direction='in')
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$k_n$', self.font_x)
        plt.xlim(np.min(self.Math2), np.max(self.Math2))
        plt.title('(b) Math2', fontsize=11, family='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.6, hspace=None)
        plt.savefig('keyw_num.png', dpi=600)
        plt.show()

    def aIrtSlop(self):
        self.Math1 = [1,2,3,4,5,6,7]
        self.Math1_RMSE = [0.4145,0.4276,0.417,0.4201,0.409,0.4161,0.4246]
        self.Math1_MAE = [0.3528,0.3781,0.3619,0.3615,0.3438,0.3529,0.3662]

        self.Math2  = [1,2,3,4,5,6,7]
        self.Math2_RMSE = [0.4678,0.4638,0.442,0.4516,0.4738,0.4526,0.4594]
        self.Math2_MAE = [0.3972,0.4008,0.3721,0.3814,0.4039,0.3843,0.392]  # 创建图并命名
        fig = plt.figure(figsize=(9, 3), facecolor='w')
        ax = fig.add_subplot(121)
        lns1 = ax.plot(self.Math1, self.Math1_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.40, 0.430)
        plt.yticks([0.40, (0.40+(0.40+0.430) / 2)/2,(0.40+0.430) / 2,((0.40+0.430) / 2+0.430)/2, 0.430])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math1, self.Math1_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.34, 0.39)
        plt.yticks([0.34, (0.34+(0.34+0.39) / 2)/2,(0.34+0.39) / 2,  ((0.34+0.39) / 2+0.39)/2,0.39])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        ax.tick_params(direction='in')
        plt.xticks([1,2,3,4,5,6,7],[0.5,1,1.2,1.5,1.7,2,2.5])
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$a$', self.font_x)
        plt.xlim(np.min(self.Math1), np.max(self.Math1))
        plt.title('(a) Math1', fontsize=11)

        ax = fig.add_subplot(122)
        lns1 = ax.plot(self.Math2, self.Math2_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.43, 0.48)
        plt.yticks([0.43, (0.43 + (0.43 + 0.48) / 2) / 2, (0.43 + 0.48) / 2, ((0.43 + 0.48) / 2 + 0.48) / 2, 0.48])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math2, self.Math2_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.36, 0.41)
        plt.yticks([0.36, (0.36+(0.36+0.41) / 2)/2,(0.36+0.41) / 2,  ((0.36+0.41) / 2+0.41)/2,0.41])
        plt.xticks([1,2,3,4,5,6,7],[0.1,0.3,0.5,1,1.2,1.5,1.7])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs,loc='lower right')
        ax.tick_params(direction='in')
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$a$', self.font_x)
        plt.xlim(np.min(self.Math2), np.max(self.Math2))
        plt.title('(b) Math2', fontsize=11)

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.6, hspace=None)
        plt.savefig('a.png', dpi=600)
        plt.show()

    def hidden_dim(self):
        self.Math1 = [1,2,3,4,5,6,7]
        self.Math1_RMSE = [0.4269,0.4244,0.4179,0.409,0.4202,0.4111,0.415]
        self.Math1_MAE = [0.3642,0.3657,0.3671,0.3438,0.3588,0.3484,0.3603]

        self.Math2  = [1,2,3,4,5,6,7]
        self.Math2_RMSE = [0.4696,0.4501,0.442,0.4469,0.4656,0.4444,0.4624]
        self.Math2_MAE = [0.3876,0.3844,0.3721,0.3795,0.3909,0.3741,0.385]
        fig = plt.figure(figsize=(9, 3), facecolor='w')
        ax = fig.add_subplot(121)
        lns1 = ax.plot(self.Math1, self.Math1_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.40, 0.43)
        plt.yticks([0.40, (0.40+(0.40+0.43) / 2)/2,(0.40+0.43) / 2,((0.40+0.43) / 2+0.43)/2, 0.43])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math1, self.Math1_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.34, 0.37)
        plt.yticks([0.34, (0.34+(0.34+0.37) / 2)/2,(0.34+0.37) / 2,  ((0.34+0.37) / 2+0.37)/2,0.37])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        ax.tick_params(direction='in')
        plt.xticks([1,2,3,4,5,6,7],[5, 10, 15, 20,25, 30,35])
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$h$', self.font_x)
        plt.xlim(np.min(self.Math1), np.max(self.Math1))
        plt.title('(a) Math1', fontsize=11, family='Times New Roman')

        ax = fig.add_subplot(122)
        lns1 = ax.plot(self.Math2, self.Math2_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.43, 0.48)
        plt.yticks([0.43, (0.43+(0.43+0.48) / 2)/2,(0.43+0.48) / 2,((0.43+0.48) / 2+0.48)/2, 0.48])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math2, self.Math2_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.36, 0.40)
        plt.yticks([0.36, (0.36+(0.36+0.40) / 2)/2,(0.36+0.40) / 2,  ((0.36+0.40) / 2+0.40)/2,0.40])
        plt.xticks([1,2,3,4,5,6,7],[3,5,10, 15,20,25,30])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs,loc='best')
        ax.tick_params(direction='in')
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$h$', self.font_x)
        plt.xlim(np.min(self.Math2), np.max(self.Math2))
        plt.title('(b) Math2', fontsize=11, family='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.6, hspace=None)
        plt.savefig('hidden_dim.png', dpi=600)
        plt.show()

    def keynum_embedingsize(self):
        self.Math = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
        self.Math1_diff = [30,30,30,30,62,62,62,62,62,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126]
        self.Math1_keynum = [22,33,44,55,66,77,88,89,110,121,132,141,151,160,169,177,183,185,187,188,188,188,188,188,188,188,188,188,188,188]
        self.Math1_embedding = [52,	63	,74	,85	,128,	139,	150	,151	,172	,247 ,258	,267,	277	,286	,295,	303	,309	,311	,313	,314,314,314,314,314,314,314,314,314,314,314 ]

        self.Math2_diff = [30, 30, 30, 30, 62, 62, 62, 62, 62, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
                           126, 126, 126, 126, 126, 126, 126, 126, 126, 126]
        self.Math2_keynum = [32,48,64,80,96,112,128,144,160,176,191,206,220,234,246,253,260,263,265,267,267,267,267,267,267,267,267,267,267,267]
        self.Math2_embedding = [62,	78,	94	,110	,158,	174	,190	,206	,222	,302 ,317,	332,	346	,360,	372	,379,	386,	389,	391,	393,	393,	393,	393,	393,	393,	393,	393,	393,	393,	393 ]

        fig = plt.figure(figsize=(9, 6), facecolor='w')
        ax = fig.add_subplot(211)
        lns1 = ax.plot(self.Math, self.Math1_diff, '*',markersize=4, c='b', lw=2.5,
                       alpha=0.8,label='difficulty dimension')

        lns2 = ax.plot(self.Math,self.Math1_keynum, 'o',markersize=4, c='r', lw=2.5,
                        alpha=0.8,         label='keywords dimension')
        lns3 = ax.plot(self.Math, self.Math1_embedding, 'v', markersize=4, c='g', lw=2.5,
                       alpha=0.8, label='embedding dimension')
        ax.tick_params(axis='y', colors='#000000', direction='in')
        plt.ylim(10, 400)
        # plt.yticks([0.39, (0.39+(0.39+0.57) / 2)/2,(0.39+0.57) / 2,((0.39+0.57) / 2+0.57)/2, 0.57])
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        ax.tick_params(direction='in')
        plt.xticks(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30])
        ax.set_ylabel('embedding dimension', fontdict=self.font_y3)
        ax.set_xlabel('the number of keywords', self.font_x)
        plt.xlim(np.min(self.Math), np.max(self.Math))
        plt.title('(a) Math1', fontsize=14, family='Times New Roman')


        ax = fig.add_subplot(212)
        lns1 = ax.plot(self.Math, self.Math2_diff, '*', markersize=4, c='b', lw=2.5,
                       alpha=0.8, label='difficulty dimension')

        lns2 = ax.plot(self.Math, self.Math2_keynum, 'o', markersize=4, c='r', lw=2.5,
                       alpha=0.8, label='keywords dimension')
        lns3 = ax.plot(self.Math, self.Math2_embedding, 'v', markersize=4, c='g', lw=2.5,
                       alpha=0.8, label='embedding dimension')

        ax.tick_params(axis='y', colors= '#000000', direction='in')
        plt.ylim(10, 450)
        # plt.yticks([0.39, (0.39+(0.39+0.57) / 2)/2,(0.39+0.57) / 2,((0.39+0.57) / 2+0.57)/2, 0.57])
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        ax.tick_params(direction='in')
        plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
        ax.set_ylabel('embedding dimension', fontdict=self.font_y3)
        ax.set_xlabel('the number of keywords', self.font_x)
        plt.xlim(np.min(self.Math), np.max(self.Math))
        plt.title('(a) Math2', fontsize=14, family='Times New Roman')


        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.95,
                            wspace=0.2, hspace=None)
        plt.savefig('keynum_embedingsize.png', dpi=600)
        # plt.show()

    def reg_l2(self):
        self.Math1 = [1,2,3,4,5,6,7]
        self.Math1_RMSE = [0.4184,0.4178,0.4203,0.409,0.4241,0.4265,0.4154]
        self.Math1_MAE = [0.363,0.3642,0.3693,0.3438,0.3561,0.3583,0.3531]

        self.Math2  = [1,2,3,4,5,6,7]
        self.Math2_RMSE = [0.4907,0.4555,0.4566,0.442,0.4574,0.4492,0.4468]
        self.Math2_MAE =[0.4178,0.3805,0.3745,0.3721,0.3868,0.3805,0.3809]

        fig = plt.figure(figsize=(9, 3), facecolor='w')
        ax = fig.add_subplot(121)
        lns1 = ax.plot(self.Math1, self.Math1_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.40, 0.430)
        plt.yticks([0.40, (0.40+(0.40+0.430) / 2)/2,(0.40+0.430) / 2,((0.40+0.430) / 2+0.430)/2, 0.430])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math1, self.Math1_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.34, 0.38)
        plt.yticks([0.34, (0.34+(0.34+0.38) / 2)/2,(0.34+0.38) / 2,  ((0.34+0.38) / 2+0.38)/2,0.38])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs,loc='lower left')
        ax.tick_params(direction='in')
        plt.xticks([1,2,3,4,5,6,7],[0.005,0.01,0.03,0.05, 0.1,0.5,1])
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$\lambda_w$', self.font_x)
        plt.xlim(np.min(self.Math1), np.max(self.Math1))
        plt.title('(a) Math1', fontsize=11, family='Times New Roman')

        ax = fig.add_subplot(122)
        lns1 = ax.plot(self.Math2, self.Math2_RMSE, 'D-', linestyle='dashed', markersize=6, c='r', lw=2.5,
                       alpha=0.8, label='RMSE')
        ax.tick_params(axis='y', colors='r', direction='in')
        plt.ylim(0.43, 0.50)
        plt.yticks([0.43, (0.43+(0.43+0.50) / 2)/2,(0.43+0.50) / 2,((0.43+0.50) / 2+0.50)/2, 0.50])
        ax2 = ax.twinx()
        lns2 = ax2.plot(self.Math2, self.Math2_MAE, '*-', linestyle='solid', markersize=10, c='b', lw=2.5,
                        alpha=0.8,
                        label='MAE')
        ax2.tick_params(axis='y', colors='b', direction='in')
        plt.ylim(0.36, 0.43)
        plt.yticks([0.36, (0.36+(0.36+0.43) / 2)/2,(0.36+0.43) / 2,  ((0.36+0.43) / 2+0.43)/2,0.43])
        plt.xticks([1,2,3,4,5,6,7],[0.005,0.01,0.03,0.05, 0.1,0.5,1])
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        ax.tick_params(direction='in')
        ax.set_ylabel('RMSE', fontdict=self.font_y1)
        ax2.set_ylabel('MAE', fontdict=self.font_y2)
        ax.set_xlabel(r'$\lambda_w$', self.font_x)
        plt.xlim(np.min(self.Math2), np.max(self.Math2))
        plt.title('(b) Math2', fontsize=11, family='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.6, hspace=None)
        plt.savefig('reg_l2.png', dpi=600)
        plt.show()



    def test_ratio(self):
        self.name=['20%','40%','60%','80%']
        fig = plt.figure(figsize=(16,  12), facecolor='w')
        self.size = 4
        self.x = np.arange(self.size)
        self.total_width, self.n =0.9, 7
        self.width = self.total_width / self.n
        self.x = self.x - (self.total_width - self.width) / 2
        self.color_list =["#FF0000","#FF7F00","yellow",'#00FF00','#00FFFF','magenta','#8B00FF','tomato','RoyalBlue','deeppink','#2E8B57']

        # Math1 RMSE *********************************************************************
        self.RMSE_IRT =[0.617033333,0.636466667,0.649566667,0.667766667]
        self.RMSE_DINA =[0.643466667,0.640366667,0.6421,0.642166667]
        self.RMSE_NMF=[0.625533333,0.6206,0.613466667,0.597633333]
        self.RMSE_PMF=[0.406066667,0.410833333,0.416333333,0.4263]
        self.RMSE_DeepFM = [0.421633333,0.420366667,0.4209,0.420966667]
        self.RMSE_DIRT = [0.430633333, 0.442866667, 0.437033333, 0.442566667]
        self.RMSE_deepCDF=[0.397633333,0.4152,0.436333333,0.465133333]


        self.std_err_IRT=[0.000731057,0.001026861,0.000405518,0.000296273]
        self.std_err_DINA = [0.001695419,0.000296273,0.00070946,0.000166667]
        self.std_err_NMF= [0.000520683,0.000360555,0.000906152,0.002306753]
        self.std_err_PMF = [0.000466667,0.000133333,0.000338296,0.000503322]
        self.std_err_DeepFM = [0.00031798,0.000120185,5.7735E-05,  0.000202759]
        self.std_err_DIRT= [0.000484195,0.00324157,0.002018525,0.001888856]
        self.std_err_deepCDF = [0.000906152,0.001537314,0.001386042,0.00116952]

        self.error_params = dict(elinewidth=1.5, ecolor=self.color_list[0], capsize=3)


        ax = fig.add_subplot(221)
        plt.bar(self.x+ 6*self.width, self.RMSE_IRT, width=self.width, label='IRT',color=self.color_list[8],edgecolor='#000000',yerr=self.std_err_IRT,error_kw=self.error_params)
        plt.bar(self.x + 5*self.width, self.RMSE_DINA, width=self.width, label='DINA',color=self.color_list[5],edgecolor='#000000',yerr=self.std_err_DINA,error_kw=self.error_params)
        plt.bar(self.x + 4 * self.width, self.RMSE_NMF, width=self.width, label='NMF',color=self.color_list[4],edgecolor='#000000',yerr=self.std_err_NMF,error_kw=self.error_params)
        plt.bar(self.x + 3 *self.width, self.RMSE_PMF, width=self.width, label='PMF',color=self.color_list[3],edgecolor='#000000',yerr=self.std_err_PMF,error_kw=self.error_params)
        plt.bar(self.x + 2*self.width, self.RMSE_DeepFM, width=self.width, label='DeepFM', color=self.color_list[2], edgecolor='#000000',yerr=self.std_err_DeepFM,error_kw=self.error_params)
        plt.bar(self.x + self.width, self.RMSE_DIRT, width=self.width, label='DIRT', color=self.color_list[1], edgecolor='#000000',yerr=self.std_err_DIRT,error_kw=self.error_params)
        plt.bar(self.x , self.RMSE_deepCDF, width=self.width, label='deepCDF',color=self.color_list[6],edgecolor='#000000',yerr=self.std_err_deepCDF,error_kw=self.error_params)
        ax.tick_params(axis='both', direction='in',labelsize=16)
        plt.xticks(np.arange(self.size),self.name)
        plt.ylim(0.30, 0.7)
        plt.yticks([0.30, 0.40,0.50,0.60,0.7])
        ax.set_ylabel('RMSE', fontdict=self.font_y3)
        plt.title('(a) Math1', fontsize=20, family='Times New Roman')


        # Math1 MAE ****************************************************************
        self.MAE_IRT =[0.406266667,0.430433333,0.447266667,0.471233333]
        self.MAE_DINA = [0.4396,0.435433333, 0.437633333,0.437766667]
        self.MAE_NMF = [0.452033333,0.453966667,0.4557,0.456433333]
        self.MAE_PMF = [0.342566667,0.345833333,0.348766667,0.353433333]
        self.MAE_DeepFM = [0.374333333,0.3724,0.3727,0.374366667]
        self.MAE_DIRT = [0.327333333, 0.3716,0.3337, 0.360533333]
        self.MAE_deepCDF = [0.345833333,0.350766667,0.3616,0.3843]

        self.std_err_IRT = [0.000906152, 0.001316983,0.000520683, 0.000384419]
        self.std_err_DINA = [0.002165641, 0.000384419, 0.000883805, 0.000218581]
        self.std_err_NMF = [0.000560753, 0.000375648, 0.000602771, 0.001098989]
        self.std_err_PMF = [0.000409607, 0.000560753, 0.001291425, 0.001065103]
        self.std_err_DeepFM = [0.000705534, 0.000458258, 0.00072111, 0.001083718]
        self.std_err_DIRT = [0.001519137, 0.02341481, 0.002267892, 0.021932345]
        self.std_err_deepCDF = [0.002420973, 0.003147662,0.001861003,0.001011599]

        ax = fig.add_subplot(223)
        plt.bar(self.x + 6 * self.width, self.MAE_IRT, width=self.width, label='IRT', color=self.color_list[8],  edgecolor='#000000', yerr=self.std_err_IRT, error_kw=self.error_params)
        plt.bar(self.x + 5 * self.width, self.MAE_DINA, width=self.width, label='DINA', color=self.color_list[5], edgecolor='#000000', yerr=self.std_err_DINA, error_kw=self.error_params)
        plt.bar(self.x + 4 * self.width, self.MAE_NMF, width=self.width, label='NMF', color=self.color_list[4], edgecolor='#000000', yerr=self.std_err_NMF, error_kw=self.error_params)
        plt.bar(self.x + 3 * self.width, self.MAE_PMF, width=self.width, label='PMF', color=self.color_list[3], edgecolor='#000000', yerr=self.std_err_PMF, error_kw=self.error_params)
        plt.bar(self.x + 2 * self.width, self.MAE_DeepFM, width=self.width, label='DeepFM', color=self.color_list[2], edgecolor='#000000',yerr=self.std_err_DeepFM, error_kw=self.error_params)
        plt.bar(self.x + self.width, self.MAE_DIRT, width=self.width, label='DIRT', color=self.color_list[1], edgecolor='#000000', yerr=self.std_err_DIRT, error_kw=self.error_params)
        plt.bar(self.x, self.MAE_deepCDF, width=self.width, label='deepCDF', color=self.color_list[6],  edgecolor='#000000', yerr=self.std_err_deepCDF, error_kw=self.error_params)
        ax.tick_params(axis='both', direction='in',labelsize=16)
        plt.xticks(np.arange(self.size), self.name)
        plt.ylim(0.30, 0.52)
        plt.yticks([0.30, 0.40, 0.50])
        ax.set_ylabel('MAE', fontdict=self.font_y3)
        ax.set_xlabel(r'test ratio', self.font_x)




        # Math2 RMSE ******************************METRICS***************************************
        self.RMSE_IRT =[0.627466667,0.640133333,0.647433333,0.656133333]
        self.RMSE_DINA =[0.652466667,0.651966667,0.652333333,0.652166667]
        self.RMSE_NMF=[0.6855,0.680433333,0.6735,0.638466667]
        self.RMSE_PMF=[0.415433333,0.417333333,0.4247,0.436733333]
        self.RMSE_DeepFM = [0.4268,0.425933333,0.425866667,0.426466667]
        self.RMSE_DIRT = [0.445366667,0.444733333,0.450533333, 0.448166667]
        self.RMSE_deepCDF=[0.422933333,0.453333333,0.477366667,0.460433333]

        self.std_err_IRT = [0.002518156, 0.000783865, 0.000895048, 0.000566667]
        self.std_err_DINA = [0.000536449, 0.000648931, 0.001964123, 0.000633333]
        self.std_err_NMF = [0.000321455, 0.000145297, 0.000351188, 0.001954766]
        self.std_err_PMF = [8.81917E-05, 8.81917E-05, 5.7735E-05, 0.000338296]
        self.std_err_DeepFM = [0.00011547, 0.000202759, 0.000120185, 0.000120185]
        self.std_err_DIRT = [0.001483614, 0.000995546, 0.002875374, 0.001185093]
        self.std_err_deepCDF = [0.000874325, 0.001322035,0.002016873, 0.001880898]

        ax = fig.add_subplot(222)
        plt.bar(self.x + 6 * self.width, self.RMSE_IRT, width=self.width, label='IRT', color=self.color_list[8],  edgecolor='#000000', yerr=self.std_err_IRT, error_kw=self.error_params)
        plt.bar(self.x + 5 * self.width, self.RMSE_DINA, width=self.width, label='DINA', color=self.color_list[5],edgecolor='#000000', yerr=self.std_err_DINA, error_kw=self.error_params)
        plt.bar(self.x + 4 * self.width, self.RMSE_NMF, width=self.width, label='NMF', color=self.color_list[4], edgecolor='#000000', yerr=self.std_err_NMF, error_kw=self.error_params)
        plt.bar(self.x + 3 * self.width, self.RMSE_PMF, width=self.width, label='PMF', color=self.color_list[3],  edgecolor='#000000', yerr=self.std_err_PMF, error_kw=self.error_params)
        plt.bar(self.x + 2 * self.width, self.RMSE_DeepFM, width=self.width, label='DeepFM', color=self.color_list[2], edgecolor='#000000', yerr=self.std_err_DeepFM, error_kw=self.error_params)
        plt.bar(self.x + self.width, self.RMSE_DIRT, width=self.width, label='DIRT', color=self.color_list[1],edgecolor='#000000', yerr=self.std_err_DIRT, error_kw=self.error_params)
        plt.bar(self.x, self.RMSE_deepCDF, width=self.width, label='deepCDF', color=self.color_list[6],  edgecolor='#000000', yerr=self.std_err_deepCDF, error_kw=self.error_params)
        ax.tick_params(axis='both', direction='in',labelsize=16)
        plt.xticks(np.arange(self.size),self.name)
        plt.ylim(0.30, 0.75)
        plt.yticks([0.30, 0.40, 0.50, 0.60, 0.7])
        ax.set_ylabel('RMSE', fontdict=self.font_y3)
        plt.title('(b) Math2', fontsize=18, family='Times New Roman')

        # Math2 MAE ****************************************************************
        self.MAE_IRT =[0.417033333,0.4335,0.4435,0.454366667]
        self.MAE_DINA = [0.4497,0.449033333,0.449533333,0.449366667]
        self.MAE_NMF = [0.527533333,0.525866667,0.525766667,0.505333333]
        self.MAE_PMF = [0.350766667,0.351,0.356666667, 0.3645]
        self.MAE_DeepFM = [0.3791,0.377866667,0.377266667,0.3808]
        self.MAE_DIRT = [0.337866667, 0.3491, 0.3538, 0.362166667]
        self.MAE_deepCDF = [0.362566667,0.384333333,0.399933333,0.386066667]

        self.std_err_IRT = [0.003186604,0.001001665,0.001167619,0.000733333]
        self.std_err_DINA = [0.00072111,0.000825295,0.002575094,0.000833333]
        self.std_err_NMF = [0.000260342,0.000218581,0.000352767,0.001476859]
        self.std_err_PMF = [0.00024037,0.000208167,0.000218581,0.000986577]
        self.std_err_DeepFM = [0.001113553,0.001462114,0.000920748,0.0001]
        self.std_err_DIRT = [0.010988226,0.00470567,0.01427165,0.009632295]
        self.std_err_deepCDF = [0.002098677,0.002331189,0.000497773,0.000785988]

        ax = fig.add_subplot(224)
        plt.bar(self.x + 6 * self.width, self.MAE_IRT, width=self.width, label='IRT', color=self.color_list[8], edgecolor='#000000', yerr=self.std_err_IRT, error_kw=self.error_params)
        plt.bar(self.x + 5 * self.width, self.MAE_DINA, width=self.width, label='DINA', color=self.color_list[5], edgecolor='#000000', yerr=self.std_err_DINA, error_kw=self.error_params)
        plt.bar(self.x + 4 * self.width, self.MAE_NMF, width=self.width, label='NMF', color=self.color_list[4],  edgecolor='#000000', yerr=self.std_err_NMF, error_kw=self.error_params)
        plt.bar(self.x + 3 * self.width, self.MAE_PMF, width=self.width, label='PMF', color=self.color_list[3], edgecolor='#000000', yerr=self.std_err_PMF, error_kw=self.error_params)
        plt.bar(self.x + 2 * self.width, self.MAE_DeepFM, width=self.width, label='DeepFM', color=self.color_list[2], edgecolor='#000000', yerr=self.std_err_DeepFM, error_kw=self.error_params)
        plt.bar(self.x + self.width, self.MAE_DIRT, width=self.width, label='DIRT', color=self.color_list[1], edgecolor='#000000', yerr=self.std_err_DIRT, error_kw=self.error_params)
        plt.bar(self.x, self.MAE_deepCDF, width=self.width, label='deepCDF', color=self.color_list[6], edgecolor='#000000', yerr=self.std_err_deepCDF, error_kw=self.error_params)
        ax.tick_params(axis='both', direction='in',labelsize=16)
        plt.xticks(np.arange(self.size), self.name)
        plt.ylim(0.30, 0.58)
        plt.yticks([0.30, 0.40, 0.50])
        ax.set_ylabel('MAE', fontdict=self.font_y3)
        ax.set_xlabel(r'test ratio', self.font_x)

        plt.legend(loc =(-1.05,-0.25),ncol=7,fontsize=18)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.95,
                            wspace=0.2, hspace=0.2)
        plt.savefig('test_ratio.png')
        # plt.show()

    def obj_sub(self):
        self.name = ['Math1-objective problems', 'Math1-subjective problems','Math2-objective problems', 'Math2-subjective problems']
        fig = plt.figure(figsize=(7, 7), facecolor='w')
        self.size = 4
        self.x = np.arange(self.size)
        self.total_width, self.n = 0.7, 7
        self.width = self.total_width / self.n
        self.x = self.x - (self.total_width - self.width) / 2
        self.color_list = ["#FF0000", "#FF7F00", "yellow", '#00FF00', '#00FFFF', 'magenta', '#8B00FF', 'tomato',
                           'RoyalBlue', 'deeppink', '#2E8B57']

        # Math1 +Math2 RMSE *********************************************************************
        self.RMSE_IRT = [0.667433333, 0.478133333,0.6556, 0.5419]
        self.RMSE_DINA = [0.6615, 0.582466667,0.665766667, 0.598666667]
        self.RMSE_NMF = [0.676033333,0.437266667,0.713866667, 0.524766667]
        self.RMSE_PMF = [0.448666667, 0.2473,0.4465, 0.3028]
        self.RMSE_DeepFM = [0.4591, 0.280366667,0.449866667, 0.3111]
        self.RMSE_DIRT = [0.4705, 0.310166667,0.467166667, 0.3299]
        self.RMSE_deepCDF = [0.4474,0.187233333,0.4634, 0.223533333]

        self.std_err_IRT = [0.00238071, 0.003080224,0.002107922, 0.001374773]
        self.std_err_DINA = [0.001644182,0.001703265,0.00136178, 0.001757207]
        self.std_err_NMF = [0.000617342, 0.001963274,0.000683943, 0.000437163]
        self.std_err_PMF = [0.000233333, 0.000458258,0.000321455, 0.000416333]
        self.std_err_DeepFM = [0.000152753, 0.000484195,0.000166667, 0.0002]
        self.std_err_DIRT = [0.005610704, 0.002903638,0.001109554, 0.010750969]
        self.std_err_deepCDF = [0.0005, 0.000425572,0.000873689, 0.00306884]

        self.error_params = dict(elinewidth=1.5, ecolor=self.color_list[0], capsize=3)

        ax = fig.add_subplot(211)
        plt.bar(self.x + 6 * self.width, self.RMSE_IRT, width=self.width, label='IRT', color=self.color_list[8],
                edgecolor='#000000', yerr=self.std_err_IRT, error_kw=self.error_params)
        plt.bar(self.x + 5 * self.width, self.RMSE_DINA, width=self.width, label='DINA', color=self.color_list[5],
                edgecolor='#000000', yerr=self.std_err_DINA, error_kw=self.error_params)
        plt.bar(self.x + 4 * self.width, self.RMSE_NMF, width=self.width, label='NMF', color=self.color_list[4],
                edgecolor='#000000', yerr=self.std_err_NMF, error_kw=self.error_params)
        plt.bar(self.x + 3 * self.width, self.RMSE_PMF, width=self.width, label='PMF', color=self.color_list[3],
                edgecolor='#000000', yerr=self.std_err_PMF, error_kw=self.error_params)
        plt.bar(self.x + 2 * self.width, self.RMSE_DeepFM, width=self.width, label='DeepFM', color=self.color_list[2],
                edgecolor='#000000', yerr=self.std_err_DeepFM, error_kw=self.error_params)
        plt.bar(self.x + self.width, self.RMSE_DIRT, width=self.width, label='DIRT', color=self.color_list[1],
                edgecolor='#000000', yerr=self.std_err_DIRT, error_kw=self.error_params)
        plt.bar(self.x, self.RMSE_deepCDF, width=self.width, label='deepCDF', color=self.color_list[6],
                edgecolor='#000000', yerr=self.std_err_deepCDF, error_kw=self.error_params)
        ax.tick_params(axis='y', direction='in')
        plt.xticks(np.arange(self.size), self.name)
        plt.ylim(0.10, 0.75)
        plt.yticks([0.20,0.30, 0.40, 0.50, 0.60, 0.7])
        ax.set_ylabel('RMSE', fontdict=self.font_y3)
        plt.title('(a) RMSE', fontsize=11, family='Times New Roman')

        # Math1+Math2 MAE ****************************************************************
        self.MAE_IRT = [0.445433333, 0.330933333,0.429833333,0.413666667]
        self.MAE_DINA = [0.437566667, 0.441566667,0.443233333,0.478333333]
        self.MAE_NMF = [0.488466667, 0.354733333,0.545166667, 0.440466667]
        self.MAE_PMF = [0.396366667, 0.184066667,0.3881,0.232733333]
        self.MAE_DeepFM = [0.4229, 0.226566667,0.409533333,0.248933333]
        self.MAE_DIRT = [0.411666667, 0.137166667,0.3681,0.1208]
        self.MAE_deepCDF = [0.395966667,0.173566667,0.399966667,0.212266667]

        self.std_err_IRT = [0.003212649, 0.002928215,0.002772083, 0.001494806]
        self.std_err_DINA = [0.002148126, 0.001991928,0.001794745, 0.002095498]
        self.std_err_NMF = [0.000185592, 0.002795433,0.000166667,0.000384419]
        self.std_err_PMF = [0.000578312, 0.000185592,0.000416333, 0.000218581]
        self.std_err_DeepFM = [0.000360555, 0.001156623,0.001017076, 0.000405518]
        self.std_err_DIRT = [0.031181743, 0.00507357,0.012070349, 0.00306159]
        self.std_err_deepCDF = [0.002325463,0.000664162,0.001507021, 0.00284273]

        ax = fig.add_subplot(212)
        plt.bar(self.x + 6 * self.width, self.MAE_IRT, width=self.width, label='IRT', color=self.color_list[8],
                edgecolor='#000000', yerr=self.std_err_IRT, error_kw=self.error_params)
        plt.bar(self.x + 5 * self.width, self.MAE_DINA, width=self.width, label='DINA', color=self.color_list[5],
                edgecolor='#000000', yerr=self.std_err_DINA, error_kw=self.error_params)
        plt.bar(self.x + 4 * self.width, self.MAE_NMF, width=self.width, label='NMF', color=self.color_list[4],
                edgecolor='#000000', yerr=self.std_err_NMF, error_kw=self.error_params)
        plt.bar(self.x + 3 * self.width, self.MAE_PMF, width=self.width, label='PMF', color=self.color_list[3],
                edgecolor='#000000', yerr=self.std_err_PMF, error_kw=self.error_params)
        plt.bar(self.x + 2 * self.width, self.MAE_DeepFM, width=self.width, label='DeepFM', color=self.color_list[2],
                edgecolor='#000000', yerr=self.std_err_DeepFM, error_kw=self.error_params)
        plt.bar(self.x + self.width, self.MAE_DIRT, width=self.width, label='DIRT', color=self.color_list[1],
                edgecolor='#000000', yerr=self.std_err_DIRT, error_kw=self.error_params)
        plt.bar(self.x, self.MAE_deepCDF, width=self.width, label='deepCDF', color=self.color_list[6],
                edgecolor='#000000', yerr=self.std_err_deepCDF, error_kw=self.error_params)
        ax.tick_params(axis='y', direction='in')
        plt.xticks(np.arange(self.size), self.name)
        plt.ylim(0.10, 0.6)
        plt.yticks([0.1,0.2,0.30, 0.40, 0.50])
        ax.set_ylabel('MAE', fontdict=self.font_y3)
        plt.title('(b) MAE', fontsize=11, family='Times New Roman')

        plt.legend(loc=(0.13, -0.45), ncol=4)
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                            wspace=0.2, hspace=0.4)
        plt.savefig('obj_sub.png', dpi=600)
        plt.show()


    def extension(self):
        self.name=['Math1-RMSE','Math1-MAE','Math2-RMSE','Math2-MAE']
        fig = plt.figure(figsize=(8, 4), facecolor='w')
        self.size = 4
        self.x = np.arange(self.size)
        self.total_width, self.n = 0.6, 7
        self.width = self.total_width / self.n
        self.x = self.x - (self.total_width - self.width) / 2
        self.color_list =["#FF0000","#FF7F00","yellow",'#00FF00','#00FFFF','magenta','RoyalBlue']

        # Math1 RMSE *********************************************************************
        self.extension_1 =[0.4130,	0.3573,0.4146,0.3590]
        self.extension_2A =[0.4382,0.3673,0.4281,0.3626]
        self.extension_2B=[0.4240,0.3584,0.4040,0.3421]
        self.extension_3=[0.4076,0.3492,0.3983,0.3339]
        self.extension_4=[0.4565,0.4164,0.4247,0.3764]
        self.extension_5=[0.4262,0.3618,0.4094,0.3339]
        self.deepCDF=[0.3968	,0.3343,0.3897,0.3217]

        ax = fig.add_subplot(111)
        plt.bar(self.x + 6 * self.width, self.extension_1, width=self.width, label='extension_1',color=self.color_list[6],edgecolor='#000000')
        plt.bar(self.x + 5 * self.width, self.extension_2A, width=self.width, label='extension_2A',color=self.color_list[5],edgecolor='#000000')
        plt.bar(self.x + 4 * self.width, self.extension_2B, width=self.width, label='extension_2B',color=self.color_list[4],edgecolor='#000000')
        plt.bar(self.x + 3 * self.width, self.extension_3, width=self.width, label='extension_3',color=self.color_list[3],edgecolor='#000000')
        plt.bar(self.x + 2 * self.width, self.extension_4, width=self.width, label='extension_4',color=self.color_list[2],edgecolor='#000000')
        plt.bar(self.x + self.width, self.extension_5, width=self.width, label='extension_5',color=self.color_list[1],edgecolor='#000000')
        plt.bar(self.x , self.deepCDF, width=self.width, label='deepCDM',color=self.color_list[0],edgecolor='#000000')
        ax.tick_params(axis='y', direction='in')
        plt.xticks(np.arange(self.size),self.name)
        plt.ylim(0.3, 0.5)
        plt.yticks([0.30, 0.35,0.40,0.45,0.50])

        plt.legend(loc =0,ncol=4)
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
        plt.savefig('extension.png', dpi=600)
        plt.show()

if __name__ == '__main__':
    p = plt_parameter_analysis()
    # p.keyw_num()
    # p.aIrtSlop()
    # p.hidden_dim()
    # p.reg_l2()
    p.test_ratio()
    # p.obj_sub()
    # p.extension()
    # p.keynum_embedingsize()
