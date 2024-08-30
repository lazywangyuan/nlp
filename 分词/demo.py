import jieba
import pkuseg
import fool

# 精确模式
seg = jieba.lcut("关于落实闵行区农业高质量发展的通知", cut_all=False)
print(seg)
# 全模式
seg = jieba.lcut("关于落实闵行区农业高质量发展的通知", cut_all=True)
print(seg)
# 搜索引擎模式
seg = jieba.lcut_for_search("关于落实闵行区农业高质量发展的通知")
print(seg)
seg = pkuseg.pkuseg()  # 以默认配置加载模型
text = seg.cut('关于落实闵行区农业高质量发展和农村集体经济发展扶持政策的通知')  # 进行分词
print(text)
text = "关于落实闵行区农业高质量发展和农村集体经济发展扶持政策的通知"
print(fool.cut(text))


def getWordDic():
    """
    读取词典文件
    载入词典
    :return:
    """
    words_dic = []
    with open("dic.txt", "r") as dic_input:
        for word in dic_input:
            words_dic.append(word.split(' ')[1].replace('\n', ''))
    return words_dic


# 实现正向匹配算法中的切词方法
def cut_words(raw_sentence, words_dic):
    # 统计词典中最长的词
    max_length = max(len(word) for word in words_dic)
    sentence = raw_sentence.strip()
    # 统计序列长度
    words_length = len(sentence)
    # 存储切分好的词语
    cut_word_list = []
    while words_length > 0:
        max_cut_length = min(max_length, words_length)
        subSentence = sentence[0: max_cut_length]
        while max_cut_length > 0:
            if subSentence in words_dic:
                cut_word_list.append(subSentence)
                break
            elif max_cut_length == 1:
                cut_word_list.append(subSentence)
                break
            else:
                max_cut_length = max_cut_length - 1
                subSentence = subSentence[0:max_cut_length]
        sentence = sentence[max_cut_length:]
        words_length = words_length - max_cut_length
    # words = "/".join(cut_word_list)
    return cut_word_list


print('最大匹配算法')
print(cut_words('关于落实闵行区农业高质量发展和农村集体经济发展扶持政策的通知', getWordDic()))


# 实现逆向最大匹配算法中的切词方法
def cut_words(raw_sentence, words_dic):
    # 统计词典中词的最长长度
    max_length = max(len(word) for word in words_dic)
    sentence = raw_sentence.strip()
    # 统计序列长度
    words_length = len(sentence)
    # 存储切分出来的词语
    cut_word_list = []
    # 判断是否需要继续切词
    while words_length > 0:
        max_cut_length = min(max_length, words_length)
        subSentence = sentence[-max_cut_length:]
        while max_cut_length > 0:
            if subSentence in words_dic:
                cut_word_list.append(subSentence)
                break
            elif max_cut_length == 1:
                cut_word_list.append(subSentence)
                break
            else:
                max_cut_length = max_cut_length - 1
                subSentence = subSentence[-max_cut_length:]
        sentence = sentence[0:-max_cut_length]
        words_length = words_length - max_cut_length
    cut_word_list.reverse()
    # words = "/".join(cut_word_list)
    return cut_word_list


print('逆向匹配算法')
print(cut_words('关于落实闵行区农业高质量发展和农村集体经济发展扶持政策的通知', getWordDic()))
import FMM
import BMM


# 实现双向匹配算法中的切词方法
def doubleMax(text, path):
    left = leftMax(path)
    right = rightMax(path)

    leftMatch = left.cut(text)
    rightMatch = right.cut(text)

    # 返回分词数较少者
    if (len(leftMatch) != len(rightMatch)):
        if (len(leftMatch) < len(rightMatch)):
            return leftMatch
        else:
            return rightMatcht
    else:  # 若分词数量相同，进一步判断
        leftsingle = 0
        rightsingle = 0
        isEqual = True  # 用以标志结果是否相同
        for i in range(len(leftMatch)):
            if (leftMatch[i] != rightMatch[i]):
                isEqual = False
            # 统计单字数
            if (len(leftMatch[i]) == 1):
                leftsingle += 1
            if (len(rightMatch[i]) == 1):
                rightsingle += 1
        if (isEqual):
            return leftMatch
        if (leftsingle < rightsingle):
            return leftMatch
        else:
            return rightMatch


print('双向匹配算法')
print(doubleMax('关于落实闵行区农业高质量发展和农村集体经济发展扶持政策的通知', getWordDic()))
# if __name__ == '__main__':
#     # 加载词典
#     dic = load_dictionary()
#     print(forward_segment('就读北京大学', dic))
# ['占用', '城市道路', '许可证']
# a = {'forwardMaxModel': ['占用，城市道路，许可证'],
#      'reverseMaxModel': ['占用，城市道路，许可证'],
#      'bidMaxModel': ['占用，城市道路，许可证'],
#      'accurateModel': ['占用，城市道路，许可证'],
#      'allModel': ['占用，城市道路，许可证'],
#      'searchModel': ['占用，城市道路，许可证']}
[{'homoionym': '消息', 'confidence': '1'},
 {'homoionym': '资讯', 'confidence': '1'},
 {'homoionym': '情报', 'confidence': '1'}]