from faster_whisper import WhisperModel
from opencc import OpenCC
import time

# 记录开始时间
start_time = time.time()

def traditional_to_simplified(traditional_text):
    cc = OpenCC('t2s')  # 't2s' 表示从繁体到简体的转换
    simplified_text = cc.convert(traditional_text)
    return simplified_text

model = WhisperModel(r"D:\models\faster-whisper-small",device='cpu')
end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
print(f"模型加载时间代码执行时间：{elapsed_time}秒")
start_time = time.time()
segments, info = model.transcribe("20char.mp3",language='zh')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"加载数据代码执行时间：{elapsed_time}秒")
start_time = time.time()
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"结束代码执行时间：{elapsed_time}秒")
traditional_text = str(segment.text)
simplified_text = traditional_to_simplified(traditional_text)
print(simplified_text)