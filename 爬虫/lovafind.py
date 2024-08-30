# https://www.ickd.cn/outlets/index.html
# 任务说明 拉取地址，当作区域识别的补充数据集
import requests
from lxml import etree
import re
import chardet
import time
import pandas as pd

res_list = []
for i in range(1, 5000):
    # time.sleep(5)
    try:
        url = 'https://www.ickd.cn/outlets/index_{0}.html'.format(i)
        response = requests.get(url)
        response.encoding = chardet.detect(response.content)['encoding']
        result = etree.HTML(response.text)
        address = result.xpath('//*[@id="net-list-left"]/div/p/text()[1]')  # 地址
        for i in address:
            print(i)
            add = re.findall('地址：(.*)', i)
            if len(add) > 0:
                print(add[0])
                res_list.append(add[0])
    except:
        res_list.append('')
pd.DataFrame({'text': res_list, 'label': res_list}).to_excel('lovefind.xlsx',index=False)

print(res_list)
