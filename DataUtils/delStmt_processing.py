import pandas as pd
#single-process:

def delStmt_file(proj,group):
    #读取raw_data
    # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\test_data_6x\\"+proj+"_result.csv",header=None,names =['label_class','type','oldname','newname','oldStmt', 'newStmt'])
    # print(data_df.head())

    # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\test_data_6x\\"+proj+"_result_change.csv",header=0)
    #extend
    data_df=pd.read_csv("C:\\project\\Projects_50\\VersionDB\\raw_data\\changeNum\\"+proj+"_merge_method.csv",header=0)



    #生成处理好后的data
    #1.去掉空方法体
    #2.去掉oldStmt=newStmt

    #新增一列记录：把方法名字去掉
    def delName(x):
        index=x.find("{")
        if index!=-1:
            x=x[index:]
        return x

    data_df['oldStmt_body'] = data_df["oldStmt"].apply(lambda x: delName(x))
    data_df['newStmt_body'] = data_df["newStmt"].apply(lambda x: delName(x))

    #
    #查看原来的label分布
    print("【origin method】 \n",data_df['label_class'].value_counts())

    #查看 空方法体的label分布
    print("【Empty method】 \n",data_df.loc[data_df['oldStmt_body']=="{}"]['label_class'].value_counts())
    print("【Empty method】 \n",data_df.loc[data_df['newStmt_body']=="{}"]['label_class'].value_counts())
    data_df.drop(data_df[data_df['oldStmt_body']=="{}"].index, inplace=True)
    data_df.drop(data_df[data_df['newStmt_body']=="{}"].index, inplace=True)
    #查看 相同stmt的label分布，一般全是label=0
    print("【same stmt】 \n",data_df.loc[data_df['oldStmt_body'] == data_df['newStmt_body']]['label_class'].value_counts())
    # data_df.drop(data_df[data_df['oldStmt_body'] == data_df['newStmt_body']].index, inplace=True)


    data_df.reset_index(drop=True, inplace=True)



    #第一次changeNum（未refine）
    print("没有变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] == 0]['label_class'].value_counts())
    print("变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] > 0]['label_class'].value_counts())


    def findNode(x):
        edges = x.split("|")
        # <x,x>,<x,x>.....
        nodeset = set()
        for e in edges:
            index = e.find(',')
            # 实体node
            node = e[index + 1:-1].strip()
            if len(node) > 0:
                nodeset.add(node)
        return nodeset
        # if score!=None:


    def refineChangeNum(x, index):

        # 假设samestmt就是changeNum=0
        changeNum = data_df.iloc[index]['changeNum']
        newStmt = data_df.iloc[index]['newStmt_body']
        if x == newStmt:
            changeNum = 0

        # 又对于old edg和new edge有变化的changeNum+
        else:
            oldedge = data_df.iloc[index]['edge']
            newedge = data_df.iloc[index]['newedge']

            oldnode = findNode(oldedge)
            newnode = findNode(newedge)
            #         print(oldnode)
            diffEnt = 0
            for n in newnode:
                if n not in oldnode:
                    diffEnt = diffEnt + 1


            if diffEnt > 0:
                changeNum = diffEnt
            elif diffEnt==0:
                changeNum=0

        return changeNum

    # print("len============", len(data_df['oldStmt_body'].to_list()))
    #
    # for x in data_df['oldStmt_body'].to_list():
    #     print(data_df.loc[data_df['oldStmt_body'] == x].index[0])
    data_df['changeNum'] = data_df["oldStmt_body"].apply(
        lambda x: refineChangeNum(x, data_df.loc[data_df['oldStmt_body'] == x].index[0]))


    print("refineChangeNum之后的分布：")
    print(data_df['label_class'].value_counts())
    print("没有变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] == 0]['label_class'].value_counts())
    print("变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] > 0]['label_class'].value_counts())

    # %%

    data_df.to_csv(
        "C:\\project\\Projects_50\\VersionDB\\raw_data\\changeNum\\"+proj+"_merge_method_refine_same.csv")
# proj="abdera"
# delStmt_file(proj)

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
        delStmt_file(str(file).strip(),group)

