import pandas as pd

data = pd.read_csv(r'F:\Pycharm\CognitiveDiagnosis\Data\Math1\data.txt', sep='	', header=None)

data = data.T
print(data)

data_des = data.describe()
print(data_des)

data_mode = data.mode()      # 众数
print(data_mode)
data_mode.index = ['mode0', 'mode1']

data_des_mode = pd.concat([data_des, data_mode])
print(data_des_mode)

data_des_mode.to_excel(r'F:\Pycharm\CognitiveDiagnosis\Data\Math1\data_des.xlsx')


