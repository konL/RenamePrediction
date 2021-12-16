import logging
import subprocess
import time
import urllib.request


import requests
import pymysql.cursors
from bs4 import BeautifulSoup
import json
# # def get_effect_data(data):
# #     results =""
# #     soup = BeautifulSoup(data, 'html.parser')
# #     projects = soup.find_all('div', class_='f4 text-normal')
# #
# #
# #     for project in projects:
# #
# #
# #         project_url = project.find('a', attrs={'class': 'v-align-middle'})['href'].strip()
# #         # print(project_url)
# #         data = project.find('a', attrs={'class': 'v-align-middle'})['data-hydro-click'].strip()
# #         dict=json.loads(data)
# #         # res=json.loads(dict['payload'])
# #         rank=dict['payload']['result_position']
# #         if rank==1:
# #             results=project_url
# #             break
# #
# #         # print("position=", dict['result_position'])
# #         # {"event_type": "search_result.click",
# #         #  "payload": {"page_number": 1, "per_page": 10, "query": "codeBERT", "result_position": 10,
# #         #              "click_id": 346950917,
# #         #              "result": {"id": 346950917, "global_relay_id": "MDEwOlJlcG9zaXRvcnkzNDY5NTA5MTc=",
# #         #                         "model_name": "Repository", "url": "https://github.com/colabyh/codebert"},
# #         #              "originating_url": "https://github.com/search?q=codeBERT", "user_id": null}}
# #
# #         # project_language = project.find('div', attrs={'class': 'd-table-cell col-2 text-gray pt-2'}).get_text().strip()
# #         # project_starts = project.find('a', attrs={'class': 'muted-link'}).get_text().strip()
# #         # update_desc = project.find('p', attrs={'class': 'f6 text-gray mb-0 mt-2'}).get_text().strip()
# #         #
# #         # result = (writer_project.split('/')[1], writer_project.split('/')[2], project_language, project_starts, update_desc)
# #         # results.append(result)
# #     return results
# #
# #
# def get_response_data(search_content):
#     # # request_url = 'https://github.com/search'
#     # request_url = 'https://search.gitee.com/search'
#     #
#     # # params = {'o': 'desc', 'q': 'python', 's': 'stars', 'type': 'Repositories', 'p': page}
#     # params = {'q': 'codeBERT','type' : 'repository'}
#     # resp = requests.get(request_url, params)
#     #
#     # return resp.text
#
#     # try:
#     user_agent = '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 Safari/537.36"'
#     headers = {'User-Agent': user_agent}
#     url = 'https://github.com/search?q='+str(search_content)
#     print(url)
#     # response = requests.get(url=url)
#     # response.encoding = 'utf-8'
#     # return response.text
#         # 构建请求的request
#     token = "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"
#     # maxTryNum=2
#
#     request = urllib.request.Request(url)
#     response = urllib.request.urlopen(request)
#
#             # 将页面转化为UTF-8编码
#     page = response.read().decode('utf-8')
#
#
#     # request.add_header('Authorization', 'token %s' % token)
#     # request.add_header(headers)
#
#         # 利用urlopen获取页面代码
#
#     return page
# #
# #     # except urllib.error.URLError:
# #
# #
# # def insert_datas(data):
# #     connection = pymysql.connect(host='localhost',
# #                                  user='root',
# #                                  password='root',
# #                                  db='test',
# #                                  charset='utf8mb4',
# #                                  cursorclass=pymysql.cursors.DictCursor)
# #     try:
# #         with connection.cursor() as cursor:
# #             sql = 'insert into project_info(project_writer, project_name, project_language, project_starts, update_desc) VALUES (%s, %s, %s, %s, %s)'
# #             cursor.executemany(sql, data)
# #             connection.commit()
# #     except:
# #         connection.close()
#
# import os
# import pandas as pd
#
# cat="Android"
# files=os.listdir('c:/project/Projects_50/'+cat+'/')
# filename='C:\\project\\Projects_50\\'+cat+'.csv'
# df = pd.read_csv(filename, header=None)
# notfound_proj=[]
# for search_content in df[0].to_list():
#
#     d=search_content.index('_')
#         #拼接url
#
#     proj=search_content[d+1:]
#
#     if proj in files:
#         pass
#     else:
#         print(proj)
#


user_agent = '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 Safari/537.36"'
headers = {'User-Agent': user_agent}

url = 'https://projects.apache.org/committees.html'

request = urllib.request.Request(url)
response = urllib.request.urlopen(request)

page = response.read().decode('utf-8')
print(page)

