# 如下，一个数据表docu_set中有三篇文章的,d1,d2,d3,如下
docu_set = {'d1': 'i love shanghai',
            'd2': 'i am from shanghai now i study in tongji university',
            'd3': 'i am from lanzhou now i study in lanzhou university of science  and  technolgy', }

docu_set = {'机构名称_值': '司法局 公安局 统计局',
            '机构地址_值': '衢州市柯城区仙霞路与花园东大道交叉路口往东北约50米 浙江省宁波市江北区人民路408附近',
            '办公时间_值': '08:30-17:30 9:00-17:00',
            '机构地址_列': '在哪 地址 位置',
            '城市名_列': '城市名 城市',
            '机构名称_列': '机构名称',
            '办公时间_列': '办公时间',
            '联系方式_列': '联系方式',
            '领导人_列': '领导人',
            '机构职能_列': '机构职能', }
# 下面用这张表做一个简单的搜索引擎，采用倒排索引
# 首先对所有文档做分词，得到文章的词向量集合
all_words = []
for i in docu_set.values():
    #    cut = jieba.cut(i)
    cut = i.split()
    all_words.extend(cut)

set_all_words = set(all_words)
print(set_all_words)
# 构建倒排索引
invert_index = dict()
for b in set_all_words:
    temp = []
    for j in docu_set.keys():
        field = docu_set[j]
        split_field = field.split()
        if b in split_field:
            temp.append(j)
    invert_index[b] = temp
find_relation = []
find_clo = []
for i in ['司法局', '位置', 'wo']:
    for j in invert_index.get(i, '无'):
        if j.find('_值') != -1:
            find_relation.append((j.replace('_值', ''), i))
        if j.find('_列') != -1:
            find_clo.append(j.replace('_列', ''))

print(find_relation)
print(find_clo)
