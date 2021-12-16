import pandas as pd

def mask(proj):
    # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\train_data_6x\\"+proj+"_prepocessed.csv")
    data_df=pd.read_csv("C:\\project\\Projects_50\\VersionDB\\raw_data\\changeNum\\"+proj+"_merge_method_refine_same.csv")
    oldname=data_df['oldname'].tolist()
    newname=data_df['newname'].tolist()
    #多处理一个步骤：把方法名字去掉
    def delName(x,index,isOld):
        # index=x.find("{")
        # if index!=-1:
        #     x=x[index:]
        # name=data_df.iloc[index]['oldname']
        if isOld:
            name=data_df.iloc[index]['oldname']
        else:
            name = data_df.iloc[index]['newname']
        x=x.replace(name,"_",1)
        print(x)




        return x

    data_df['oldStmt'] = data_df["oldStmt"].apply(lambda x: delName(x,data_df.loc[data_df['oldStmt']==x].index[0],True))
    data_df['newStmt'] = data_df["newStmt"].apply(lambda x: delName(x,data_df.loc[data_df['newStmt']==x].index[0],False))


    # data_df.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\"+proj+"_test_mask_larger.csv")
    data_df.to_csv("C:\\project\\Projects_50\\VersionDB\\process_data\\test_data\\"+proj+"_method_same.csv")
    print(data_df['label_class'].value_counts())
    print("没有变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] == 0]['label_class'].value_counts())
    print("变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] > 0]['label_class'].value_counts())

# proj = "abdera"
# mask(proj)

import os
path="C:\\project\\Projects_50\\VersionDB\\raw_project\\GROUP_1"
allfilelist = os.listdir(path)
group="1"
# 遍历该文件夹下的所有目录或者文件
for file in allfilelist:
    if file=="abdera" or file=="101repo":
        pass
    else:
        print(str(file))
        mask(str(file).strip())