import edge_tts
import asyncio

# 入伍登记表
# 会计专业继续教育查询
# 川沙新镇社区教育中心英语基础班
# 上海市静安区公共服务中心电话号码是多少呢
TEXT = '你认为你最大的优点是什么？'
voice = 'zh-CN-YunxiNeural'
output = 'start.mp3'


async def my_function():
    tts = edge_tts.Communicate(text=TEXT, voice=voice)
    await tts.save(output)


if __name__ == '__main__':
    asyncio.run(my_function())
