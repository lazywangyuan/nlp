import re


def split_text(text, max_length):
    """
    切割文本，确保每个段落在规定字数内，句子完整。

    :param text: 原始文本
    :param max_length: 每个段落的最大字数
    :return: 切割后的段落列表
    """
    # 使用正则表达式分割句子，以句号或换行符为分隔符
    sentences = re.split(r'[。！？\n]', text)

    # 清洗空格和去除多余的换行符
    sentences = [s.strip() for s in sentences if s.strip()]

    paragraphs = []  # 存储最终的段落列表
    current_paragraph = []  # 当前段落的句子列表

    for sentence in sentences:
        # 如果当前段落加上新句子不超过最大长度，加入当前句子
        if sum(len(s) for s in current_paragraph) + len(sentence) <= max_length:
            current_paragraph.append(sentence)
        else:
            # 当前段落已满，保存到段落列表，并开始新的段落
            if current_paragraph:
                paragraphs.append(''.join(current_paragraph))
                current_paragraph = [sentence]

    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(''.join(current_paragraph))

    return paragraphs


# 示例文本
text = "这是一个示例文本。它包含了多个句子。每个句子都可能很长，也可能很短。我们需要确保每个段落在规定字数内切割。"

# 调用函数，设置最大长度为50个字符
max_length = 50
paragraphs = split_text(text, max_length)

# 打印结果
for i, p in enumerate(paragraphs):
    print(f"段落 {i + 1}: {p}")