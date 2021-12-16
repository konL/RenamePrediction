

from subprocess import check_call
import pandas as pd
import os
if __name__ == '__main__':
    cat="Android"


    # 要下载的项目
    filename='C:\\project\\Projects_50\\'+cat+'.csv'
    df = pd.read_csv(filename, header=None)
    notfound_proj=[]
    for search_content in df[0].to_list():
        print(search_content)
        d=search_content.index('_')
        #拼接url
        user=search_content[:d]
        proj=search_content[d+1:]

        data="/"+user+"/"+proj
        proj_url="git@github.com:"+data+".git"
        print(proj_url)
        # if proj_url=="":
        #     continue
        print('git clone '+proj_url+' c:/project/Projects_50/'+cat+'/'++search_content)
        #执行git命令
        try:
            check_call('git clone '+proj_url+' c:/project/Projects_50/'+cat+'/'+proj, shell=True)
        except Exception as e:
            print(Exception, ',', e)
            notfound_proj.append(search_content)
            continue
#打印找不到的项目
print(notfound_proj)

with open("c:/project/Projects_50/"+cat+"/nofound.txt","w") as f:
    f.write(str(notfound_proj))
