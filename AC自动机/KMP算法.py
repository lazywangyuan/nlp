# 计算next数组
def get_next(s):
    size = len(s)
    next = list(range(size))
    next[0] = -1
    k = -1
    j = 0
    # 计算next[j]时，考察的字符串是模式串的前j-1个字符，求其中最大相等的前缀和后缀的数量，与s[j]无关
    while j < size - 1:
        # s[k]表示前缀，s[j]表示后缀
        # k==-1表示未找到k前缀和k后缀相等
        if k == -1 or s[j] == s[k]:
            j += 1
            k += 1
            next[j] = k
        else:
            k = next[k]
    return next


# 计算next数组优化
def get_new_next(s):
    size = len(s)
    next = list(range(size))
    next[0] = -1
    k = -1
    j = 0
    # 计算next[j]时，考察的字符串是模式串的前j-1个字符，求其中最大相等的前缀和后缀的数量，与s[j]无关
    while j < size - 1:
        # s[k]表示前缀，s[j]表示后缀
        # k==-1表示未找到k前缀和k后缀相等
        if k == -1 or s[j] == s[k]:
            j += 1
            k += 1
            if s[j] == s[k]:
                # 特例优化，在和大字符串做匹配的时候
                # 如果s[j]!=大字符串节点的值，则要把s向前滑动到s[k]和大字符串节点做比较
                # 如果s[j] == s[k]，则s[k]也!=大字符串节点的值
                # 这时就需要再次滑动到s[next[k]]做比较
                # 为了减少中间不必要的滑动，直接让s滑动到s[next[k]]就好
                next[j] = next[k]
            else:
                next[j] = k
        else:
            k = next[k]
    return next


# 从文本串ss中找出模式串s第一次出现的位置
def kmp(ss, s):
    ss_len = len(ss)
    s_len = len(s)
    s_next = get_new_next(s)  # 用方法get_next也是可以的，其他代码不变
    i = 0  # ss比较的位置
    j = 0  # s比较的位置
    result = -1  # 结果
    while i < ss_len:
        if j == -1 or ss[i] == s[j]:
            j += 1
            i += 1
        else:
            j = s_next[j]
        if j == s_len:
            result = i - s_len
            break
    return result


if __name__ == '__main__':
    ss = '市二医院'
    s = '市二医院在那'
    next = get_next(s)
    print('计算next数组结果：')
    print(next)
    next = get_new_next(s)
    print('计算next数组优化结果：')
    print(next)
    result = kmp(ss, s)
    print('文本串"%s"第一次出现"%s"的位置:%d' % (ss, s, result))
