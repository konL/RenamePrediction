import pandas as pd

def mask(proj):
    #test数据（createchange之后）
    data_df=pd.read_csv("C:\\project\\MethodPrediction_All_ID\\raw_data\\changeNum\\"+proj+"_merge_refine.csv")
    oldname=data_df['oldname'].tolist()
    newname=data_df['newname'].tolist()
    #多处理一个步骤：把方法名字去掉
    def delName(x,index,isOld):
        if isOld:
            name=data_df.iloc[index]['oldname']
        else:
            name = data_df.iloc[index]['newname']
        x=x.replace(name,"_",1)
        print(x)




        return x

    data_df['oldStmt'] = data_df["oldStmt"].apply(lambda x: delName(x,data_df.loc[data_df['oldStmt']==x].index[0],True))
    data_df['newStmt'] = data_df["newStmt"].apply(lambda x: delName(x,data_df.loc[data_df['newStmt']==x].index[0],False))
    print(data_df['label_class'].value_counts())
    print("没有变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] ==0]['label_class'].value_counts())
    print("变化过的类别分布：")
    print(data_df.loc[data_df['changeNum'] >0]['label_class'].value_counts())


    # data_df.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\"+proj+"_test_mask_larger.csv")
    data_df.to_csv("C:\\project\\MethodPrediction_All_ID\\process_data\\test_data\\"+proj+"_test.csv")

proj="beam"
mask(proj)