#做统计
import pandas as pd
proj="dubbo"

#为了防止出现 Error tokenizing data.，加上delimeter
# data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\test_data_6x\\"+proj+"_result_t.csv",delimiter=",",header=None,names =['label_class','type','oldname','newname','oldStmt', 'newStmt','edge'])
#extend

data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data_extend\\context\\OK\\"+proj+"_merge.csv",encoding='unicode_escape',delimiter=",",header=None)
data_df.columns = ["label_class", "type", "oldname", "newname", "oldStmt", "newStmt",  "edge"]
print(data_df[6])
# data_df
# data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\changeIdentifier\\zeppelin_result.csv",header=0)

# 建立一个word_Index 单词：label
word_index = {}
changeFile = pd.read_csv("C:\\project\\IdentifierStyle\\log\\dump\\" + proj + ".csv", delimiter=",",header=None,error_bad_lines=False)
for indexs in changeFile.index:
    ent = changeFile.loc[indexs].values[3].split('<-')
    # 最后会有空的
    oneent = ent[len(ent) - 2]
    print(oneent)

    #     print(type(ent),type(str(label)))
    word_index.update({oneent.strip(): 1})

# changeFile
print(len(word_index))
print(len(changeFile))


def process(x):
    ent = x.split('<-')
    #最后一个，最旧的名字
    oneent = ent[len(ent) - 2]
    return oneent


nameSet = changeFile[3].apply(lambda x: process(x)).tolist()
print(len(nameSet))


# 这里要实现获得当前处理标识符的index，index之前根据edge找
# def cal(x):
#     edges=x.split("|")
#     node=set()
#     changeEnt=0
#     sumEnt=len(edges)
#     for e in edges:
#         index=e.find(',')
#         node=e[index+1:-1].strip()
#         score=word_index.get(node)
#         if score!=None:
#             print(node,changeEnt)
#             changeEnt=changeEnt+word_index.get(node)
#         else:
#             print(node,changeEnt)
#             changeEnt=changeEnt+0

# 实际整个相关都是oldname相关的实体


def find_change(index, changeFile):
    change_relate_Ent = {}
    print(index)
    for i in range(index, len(changeFile)):
        ent = changeFile.loc[i].values[3].split('<-')
        oneent = ent[len(ent) - 2]
        change_relate_Ent.update({oneent.strip(): 1})
    return change_relate_Ent



def cal(x):
    edges = x.split("|")
    node = set()
    changeEnt = 0
    sumEnt = len(edges)
    index = edges[0].find(',')
    name = edges[0][1:index].strip()
    if word_index.get(name) != None:
        # label=1
        #         cur_index=len(nameSet) - 1 - nameSet[::-1].index(name)
        cur_index = nameSet.index(name)
    #         print("label=1")

    else:
        cur_index = 0
    #         print("label=0")
    # 在proj.csv中，index后的相关实体修改是已知的

    change_relate_Ent = find_change(cur_index, changeFile)
    if cur_index == 0:
        print("label=0", len(change_relate_Ent))
    else:
        print("label=1", len(change_relate_Ent))
    for e in edges:
        index = e.find(',')
        node = e[index + 1:-1].strip()
        score = change_relate_Ent.get(node)
        if score != None:
            changeEnt = changeEnt + 1

    #     print(changeEnt)
    #     print(sumEnt)
    return changeEnt
change_relate_Ent = find_change(0, changeFile)
# print(change_relate_Ent)
from difflib import SequenceMatcher#导入库
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()#
import Levenshtein
def cal_no_order(x):
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
        # if score!=None:
        #     print("rename pair",score,e)



        # print(similarity(node.lower(), name.lower()))
        if score != None and name!=node:
            changeEnt = changeEnt + 1




    return changeEnt


data_df["changeNum"] = data_df[6].apply(lambda x: cal_no_order(x))

#在与某个标识符相关的实体集合中，含有的实体变化个数
print("changeNum统计")
print(data_df["changeNum"].value_counts())
print("原始的类别分布：")
print(data_df[0].value_counts())
print("变化较少的类别分布：")
print(data_df.loc[data_df["changeNum"] <=0][0].value_counts())
print("变化较多的类别分布：")
print(data_df.loc[data_df["changeNum"] >0][0].value_counts())
# data_df.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data\\test_data_6x\\"+proj+"_result_change.csv")
data_df.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\raw_data_extend\\changeNum\\"+proj+"_rawdata.csv")

# data_df=pd.read_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test.csv",encoding='unicode_escape',delimiter=",",header=0)
# print("changeNum统计")
# print(data_df["changeNum"].value_counts())
# print("原始的类别分布：")
# print(data_df["label_class"].value_counts())
# print("变化较少的类别分布：")
# print(data_df.loc[data_df["changeNum"] <=0]["label_class"].value_counts())
# print("变化较多的类别分布：")
# print(data_df.loc[data_df["changeNum"] >0]["label_class"].value_counts())