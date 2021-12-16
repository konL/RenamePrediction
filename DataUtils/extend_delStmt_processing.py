
import pandas as pd


def delStmt_file(proj):
    #读取raw_data
    # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\test_data_6x\\"+proj+"_result.csv",header=None,names =['label_class','type','oldname','newname','oldStmt', 'newStmt'])
    # print(data_df.head())


    #extend
    data_df=pd.read_csv("C:\\project\\MethodPrediction_All_ID\\raw_data\\changeNum\\"+proj+"_merge.csv",header=0)
    print(data_df.head())


    #生成处理好后的data
    #1.去掉空方法体
    #2.去掉oldStmt=newStmt

    def isMatch( s: str, p: str) -> bool:
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        for j in range(2, n, 2):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 2] or dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.') \
                    if p[j - 1] == '*' else \
                    dp[i - 1][j - 1] and (p[j - 1] == '.' or s[i - 1] == p[j - 1])
        return dp[-1][-1]
    #新增一列记录：把标识符名字去掉
    #x：statement
    import re  # 第一步，要引入re模块
    def delName(x,k_index,isOld):

        classIndex=isMatch(x, '.* *class *.*{.*')
        interfaceIndex = isMatch(x, '.* *interface *.*{.*')
        enumIndex = isMatch(x, '.* *enum *.*{.*')
        assignIndex = isMatch(x, '.* *= *.*{.*')
        index = x.find("{")

        if index != -1 and ((not classIndex) and (not interfaceIndex) and (not enumIndex)and(not assignIndex) ):
            # method
            x = x[index:]
        else:
            if assignIndex:
                equidx = x.find("=")
                if x[0:equidx].find("{") != -1:
                    x = x[index:]
            else:
                if isOld:
                    name = data_df.iloc[k_index]['oldname']
                    # print(name,x)
                else:
                    name = data_df.iloc[k_index]['newname']
                x = x.replace(name, "_")


        return x

    data_df['oldStmt_body'] = data_df["oldStmt"].apply(lambda x: delName(x,data_df.loc[data_df['oldStmt']==x].index[0],True))
    data_df['newStmt_body'] = data_df["newStmt"].apply(lambda x: delName(x,data_df.loc[data_df['newStmt']==x].index[0],False))
    print(data_df.head())
    #
    #查看原来的label分布
    print("【origin method】 \n",data_df['label_class'].value_counts(),"\n ---------------------")

    # #查看 空方法体的label分布
    print("【Empty method】 \n",data_df.loc[data_df['oldStmt_body']=="{}"]['label_class'].value_counts())
    print("【Empty method】 \n",data_df.loc[data_df['newStmt_body']=="{}"]['label_class'].value_counts())
    print("【Empty method】 \n",data_df.loc[data_df['oldStmt_body']=="{"]['label_class'].value_counts())
    print("【Empty method】 \n",data_df.loc[data_df['newStmt_body']=="{"]['label_class'].value_counts())
    data_df.drop(data_df[data_df['oldStmt_body']=="{}"].index, inplace=True)
    data_df.drop(data_df[data_df['newStmt_body']=="{}"].index, inplace=True)
    data_df.drop(data_df[data_df['oldStmt_body']=="{"].index, inplace=True)
    data_df.drop(data_df[data_df['newStmt_body']=="{"].index, inplace=True)


    #查看 相同stmt的label分布，一般全是label=0
    print("【same stmt】 \n",data_df.loc[data_df['oldStmt_body'] == data_df['newStmt_body']]['label_class'].value_counts())
    data_df.drop(data_df[data_df['oldStmt_body'] == data_df['newStmt_body']].index, inplace=True)
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
            oldedge = data_df.iloc[index]['oldedge']
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
    print(data_df.loc[data_df['changeNum'] ==0]['label_class'].value_counts())
    print(data_df.loc[data_df['changeNum'] == 0]['type'].value_counts())
    print("变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] >0]['label_class'].value_counts())
    print(data_df.loc[data_df['changeNum'] > 0]['type'].value_counts())

    data_df.to_csv("C:\\project\\MethodPrediction_All_ID\\raw_data\\changeNum\\"+proj+'_merge_refine.csv')


proj="beam"
delStmt_file(proj)