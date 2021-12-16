#把每k-1个文件合并为一个train data数据集
projs=[]
import os
path="C:\\project\\Projects_50\\VersionDB\\raw_project\\GROUP_1"
allfilelist = os.listdir(path)
group="1"
# 遍历该文件夹下的所有目录或者文件
for file in allfilelist:
    projs.append(file)
print(projs)
import pandas as pd
for i in range(len(projs)):
    cur=projs[i]
    train_list = []
    data = pd.DataFrame()
    train_list.extend(projs[:i])
    train_list.extend(projs[i+1:])
    #根据train_list 读取 proj_prepocessed 再写入
    print(len(data))
    for proj in train_list:
        data_df=pd.read_csv("C:\\project\\Projects_50\\VersionDB\\process_data\\test_data\\GROUP1_same\\"+proj+"_method_same.csv",header=0)
        # data_df=data_df[['label_class','oldname','newname','oldStmt','newStmt']]

        data = pd.concat([data, data_df],sort=False)
    print(len(data))
    print(data.head())
    data.to_csv("C:\\project\\Projects_50\\VersionDB\\process_data\\train_data\\GROUP1_same\\"+cur+"_train.csv")
#


