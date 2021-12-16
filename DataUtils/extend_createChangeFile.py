#做统计
import pandas as pd

def create_Change_file(proj):
    #为了防止出现 Error tokenizing data.，加上delimeter
    data_df=pd.read_csv("C:\\project\\MethodPrediction_All_ID\\raw_data\\context\\"+proj+"_merge.csv",delimiter=",",low_memory=False,encoding='unicode_escape',header=None,names =['label_class','type','oldname','newname','oldStmt', 'newStmt','oldedge','newedge'])
    print(data_df.columns)
    # data_df
    # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\changeIdentifier\\zeppelin_result.csv",header=0)
    print(data_df['type'].head())
    # 建立一个word_Index 单词：label
    word_index = {}
    changeFile = pd.read_csv("C:\\project\\MethodPrediction_All_ID\\Renaming\\" + proj + ".csv", delimiter=",",header=None,error_bad_lines=False)
    for indexs in changeFile.index:
        ent = changeFile.loc[indexs].values[3].split('<-')
        # 最后会有空的
        oneent = ent[len(ent) - 2]


        #     print(type(ent),type(str(label)))
        word_index.update({oneent.strip(): 1})

    # changeFile
    print("word_index=", len(word_index))
    print("change file=", len(changeFile))


    def process(x):
        ent = x.split('<-')
        #最后一个，最旧的名字
        oneent = ent[len(ent) - 2]
        return oneent


    nameSet = changeFile[3].apply(lambda x: process(x)).tolist()

    def find_change(index, changeFile):
        change_relate_Ent = {}

        for i in range(index, len(changeFile)):
            ent = changeFile.loc[i].values[3].split('<-')
            oneent = ent[len(ent) - 2]
            change_relate_Ent.update({oneent.strip(): 1})
        return change_relate_Ent



    change_relate_Ent = find_change(0, changeFile)
    print(" change_relate_Ent=",len(change_relate_Ent))
    def cal_no_order(x):
        print(x)
        edges = x.split("|")
        # print(edges)
        node = set()
        changeEnt = 0
        sumEnt = len(edges)
        index = edges[0].find(',')
        name = edges[0][1:index].strip()
        # print(index,name)
        for e in edges:
            index = e.find(',')
            #实体node
            node = e[index + 1:-1].strip()
            score = change_relate_Ent.get(node)
            if score != None:
                changeEnt = changeEnt + 1


        return changeEnt

    data_df['changeNum'] = data_df["oldedge"].apply(lambda x: cal_no_order(x))

    #在与某个标识符相关的实体集合中，含有的实体变化个数
    print("changeNum统计")
    print(data_df['changeNum'].value_counts())
    print("原始的类别分布：")
    print(data_df['label_class'].value_counts())
    print("没有变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] ==0]['label_class'].value_counts())

    print("变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] >0]['label_class'].value_counts())

    data_df.to_csv("C:\\project\\MethodPrediction_All_ID\\raw_data\\changeNum\\"+proj+"_merge.csv")

proj="beam"
create_Change_file(proj)