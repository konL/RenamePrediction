import os
import  sys


def getallfile(path,allpath,allname):

    allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath=os.path.join(path,file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            getallfile(filepath,allpath,allname)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            testname = file
            if testname.endswith(".java"):
              filepath1 = filepath.replace('/','\\')
              filepath2 = filepath1.replace('\\','\\' +'\\')
              #filepath3 = filepath2.replace(' ','\\')
              allpath.append(filepath2)
              allname.append(file)
    return allpath, allname


if __name__ == "__main__":
    projectname = "camel"
    rootdir = "C:\\project\\MethodPrediction_All_ID\\data\\GitProject\\" + projectname
    file_handle = open('C:\\project\\MethodPrediction_All_ID\\data\\JavaFileIndex\\' + projectname + '.txt', mode='w', encoding='utf-8')
    files, names = getallfile(rootdir,[],[])
    print(len(files))
    for file in files:

        file_handle.write(file + '\n')
    # files = os.listdir(r'C:\\project\\Projects_50\\Java')
    # for file in files:
    #     projectname = file
    #     rootdir = "C:\\project\\Projects_50\\Java\\" + projectname
    #     file_handle = open('c:\\project\\Projects_50\\JavaFileIndex\\' + projectname +'.txt', mode='w',encoding='utf-8')
    #     files, names = getallfile(rootdir,[],[])
    #     print(len(files))
    #     for file in files:
    #
    #         file_handle.write(file + '\n')
        # break
        # print("-------------------------")



    #for name in names:
        #print(name)
