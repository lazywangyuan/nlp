import os
import requests
import pandas as pd

headers = {
    'Authorization': """"""
}


def load_data(path_url):
    url = ''
    files = {
        'file': (path_url, open(path_url, 'rb'))  # 'file'是服务器端接收文件的字段名
    }
    response = requests.post(url, files=files, headers=headers)
    print(eval(response.text)['data'])
    return eval(response.text)['data']


def predict(path_url):
    url = ''
    ori_data = load_data(path_url)
    data = {
        'data': ori_data}
    # 发送POST请求
    response = requests.post(url, data=data, headers=headers)
    json_ = eval(response.text)['data']
    complete_Disassemble = {
        "文件名称": "",
        "文件文号": "",
    }
    complete_Disassemble.update(json_)
    complete_Disassemble['常见问题'] = [complete_Disassemble['常见问题']]
    df = pd.DataFrame(complete_Disassemble)
    df['原文'] = [ori_data]
    return df


all_path = r''
df_list = []
for root, dirs, files in os.walk(all_path):
    for file in files:
        print('正确解析')
        print(file)
        try:
            # 将文件的完整路径添加到列表中
            json_df = predict(os.path.join(root, file))
            json_df['filename'] = [file]
            df_list.append(json_df)
            pd.concat(df_list).to_excel('', index=False)
        except:
            print('有错误文章')
            print(file)
