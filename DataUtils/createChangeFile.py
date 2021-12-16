#做统计
import pandas as pd

def create_Change_file(proj,group):
    #为了防止出现 Error tokenizing data.，加上delimeter
    # data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\test_data_6x\\"+proj+"_result_t.csv",delimiter=",",header=None,names =['label_class','type','oldname','newname','oldStmt', 'newStmt','edge'])
    # data_df

    data_df=pd.read_csv("C:\\project\\Projects_50\\VersionDB\\raw_data\\context\\GROUP_1\\"+proj+"_merge_method.csv",delimiter=",",header=None,encoding='unicode_escape',names =['label_class','type','oldname','newname','oldStmt', 'newStmt','edge','newedge'])


    print(data_df[['oldname','newname']].head())
    # 建立一个word_Index 单词：label
    word_index = {}
    changeFile = pd.read_csv("C:\\project\\Projects_50\\Renaming\\"+group+"\\" + proj + ".csv", delimiter=",",header=None,error_bad_lines=False)
    for indexs in changeFile.index:
        ent = changeFile.loc[indexs].values[3].split('<-')
        # 最后会有空的
        oneent = ent[len(ent) - 2]


        #     print(type(ent),type(str(label)))
        word_index.update({oneent.strip(): 1})

    # changeFile
    print("word_index=",len(word_index))
    print("change file=",len(changeFile))


    def process(x):
        ent = x.split('<-')
        #最后一个，最旧的名字
        oneent = ent[len(ent) - 2]
        return oneent


    nameSet = changeFile[3].apply(lambda x: process(x)).tolist()

    def find_change(index, changeFile):
        change_relate_Ent = {}
        print(index)
        for i in range(index, len(changeFile)):
            ent = changeFile.loc[i].values[3].split('<-')
            oneent = ent[len(ent) - 2]
            change_relate_Ent.update({oneent.strip(): 1})
        return change_relate_Ent



    change_relate_Ent = find_change(0, changeFile)
    def cal_no_order(x):

        edges = x.split("|")
        node = set()
        changeEnt = 0
        sumEnt = len(edges)
        index = edges[0].find(',')
        name = edges[0][1:index].strip()
        for e in edges:
            index = e.find(',')
            #实体node
            node = e[index + 1:-1].strip()
            score = change_relate_Ent.get(node)
            if score != None:
                changeEnt = changeEnt + 1

        #     print(changeEnt)
        #     print(sumEnt)
        return changeEnt



    data_df['changeNum'] = data_df["edge"].apply(lambda x: cal_no_order(x))

    #在与某个标识符相关的实体集合中，含有的实体变化个数
    print("类别分布：\n")
    print(data_df['changeNum'].value_counts())
    print(data_df['label_class'].value_counts())
    print("变化分布：\n")
    print(data_df.loc[data_df['changeNum'] <=0]['label_class'].value_counts())
    print(data_df.loc[data_df['changeNum'] >0]['label_class'].value_counts())


    data_df.to_csv("C:\\project\\Projects_50\\VersionDB\\raw_data\\changeNum\\"+proj+"_merge_method.csv")

#single-process:
# proj="abdera"
# create_Change_file(proj)

#batch
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
        create_Change_file(str(file).strip(),group)