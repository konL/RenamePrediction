#先获取add/delete columns
import pandas as pd
proj="101repo"

data_df=pd.read_csv("C:\\project\\Projects_50\\VersionDB\\raw_data\\changeNum\\"+proj+"_merge_method.csv",header=None,names =['proj','file','ins_ent','del_ent'])
data_df
# ,names =['label_class','type','oldname','newname','oldStmt', 'newStmt']
# # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\changeIdentifier\\zeppelin_result.csv",header=0)
# data_df[['oldname','newname']]

#删除全部为空的行数
data_df.drop(data_df[data_df['ins_ent'] .isna() &  data_df['del_ent'].isna()].index, inplace=True)



#删除不包含的文件
#1.遍历项目文件夹获取文件，不在文件夹中的文件就删除

import os
file_path = 'C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_project\\'+proj
# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中
# 输入文件夹路径、空文件列表[]
# 返回 文件列表Filelist,包含文件名（完整路径）
def get_filelist(filedir, Filelist):
    for s in os.listdir(filedir):
        Filelist.append(os.path.basename(s))
    return Filelist

l = get_filelist(file_path, [])

data_df.drop(data_df[~data_df['file'].isin(l)].index, inplace=True)
data_df

#读取context文件，在old删除del_int,在new上删除ins_ent
old_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\context_data\\"+proj+"_old_context.csv",header=None)
new_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\context_data\\"+proj+"_new_context.csv",header=None)
print(len(old_df))
print(len(new_df))
old_df.head()


# 读取dins_ent和del_ent
def toent(l):
    ent = []
    for e in l:
        if type(e) != float:
            item = e.strip().split(" ")
            ent.extend(item)
            print(e)
    print(len(set(ent)))
    return set(ent)


ins_ent = toent(data_df['ins_ent'].tolist())
del_ent = toent(data_df['del_ent'].tolist())
#根据ins_ent和del_ent删除
#读取context文件，在old删除del_int,在new上删除ins_ent
print(del_ent)

old_df.loc[~old_df[2].isin(del_ent)].to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\"+proj+'_del_old_ent.csv')
print(len(old_df))
print(ins_ent)

new_df.loc[~new_df[2].isin(ins_ent)].to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\"+proj+'_del_new_ent.csv')
print(len(new_df))