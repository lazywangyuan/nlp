#!/usr/bin/env python3
import asyncio
import edge_tts
from faster_whisper import WhisperModel
from opencc import OpenCC
import io

traditional_text = ''
TEXT = '今天是我娶妻的日子'
VOICE = 'zh-CN-YunxiNeural'
byte_stream1 = bytearray(b'')

model = WhisperModel("D:\models\whisper-large-v2", device='cpu')


async def amain() -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            byte_stream1.extend(chunk["data"])


def traditional_to_simplified(traditional_text):
    cc = OpenCC('t2s')  # 't2s' 表示从繁体到简体的转换
    simplified_text = cc.convert(traditional_text)
    return simplified_text


if __name__ == "__main__":
    loop = asyncio.get_event_loop_policy().get_event_loop()
    loop.run_until_complete(amain())
    file_like = io.BytesIO(byte_stream1)
    segments, info = model.transcribe(file_like)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        traditional_text = str(segment.text)
    if len(traditional_text) > 0:
        simplified_text = traditional_to_simplified(traditional_text)
        print(simplified_text)
