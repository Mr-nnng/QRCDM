#导入数据分析包
import pandas as pd
import matplotlib.pyplot as plt

#导入可视化包
import matplotlib.pyplot as plt

from pylab import *
data= pd.read_csv(r'F:\Pycharm\CognitiveDiagnosis\Data\Math1\data.txt', sep='	', header=None)
data2= pd.read_csv(r'F:\Pycharm\CognitiveDiagnosis\Data\Math2\data.txt', sep='	', header=None)
rcParams['axes.unicode_minus'] = False

rcParams['font.sans-serif'] = ['Simhei']


rcParams['axes.unicode_minus'] = False
rcParams['font.sans-serif'] = ['Simhei']

box_1, box_2, box_3, box_4,box_5,box_6, box_7, box_8, box_9,box_10,box_11, box_12, box_13, box_14,box_15,box_16, box_17, box_18, box_19, box_20   = data[0], data[1], data[2], data[3],data[4],data[5], data[6], data[7],data[8], data[9],data[10], data[11], data[12], data[13], data[14],data[15], data[16], data[17], data[18], data[19]
li=[box_1, box_2, box_3, box_4,box_5,box_6, box_7, box_8, box_9,box_10,box_11, box_12, box_13, box_14,box_15,box_16, box_17, box_18, box_19, box_20]

count=1
for i in li:
    print(count)
    print(i.value_counts())
    count+=1


box_1, box_2, box_3, box_4,box_5,box_6, box_7, box_8, box_9,box_10,box_11, box_12, box_13, box_14,box_15,box_16, box_17, box_18, box_19, box_20   = data2[0], data2[1], data2[2], data2[3],data2[4],data2[5], data2[6], data2[7],data2[8], data2[9],data2[10], data2[11], data2[12], data2[13], data2[14],data2[15], data2[16], data2[17], data2[18], data2[19]

li=[box_1, box_2, box_3, box_4,box_5,box_6, box_7, box_8, box_9,box_10,box_11, box_12, box_13, box_14,box_15,box_16, box_17, box_18, box_19, box_20]

count=1
for i in li:
    print(count)
    print(i.value_counts())
    count+=1






