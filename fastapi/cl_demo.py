from fastapi import FastAPI
import requests


def post(text1,text2):
    url = 'http://10.10.22.112:8010/items'

    # 构建请求数据
    data = {
        'item': text1,
        'item2': text2
    }

    # 发送 post 请求
    response = requests.post(url=url, data=data, timeout=10)

    # 检查响应状态
    if response.status_code == 200:
        print('请求成功')
        print(response.json())
    else:
        print('请求失败，状态码：', response.status_code)
post('这是一个文本','这是另一个文本')