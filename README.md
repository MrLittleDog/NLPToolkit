# NLPToolkit
本项目包含了我个人常用的一些中文自然语言处理工具.

## 依赖
* 代码在python 3.6下编写，其他必须的依赖见requirements.txt
* pyltp安装使用：https://github.com/HIT-SCIR/pyltp

## 功能说明
* preprocess.py
|文件操作|去噪操作|其他|
|:----:|:----:|:---:|
|读写文本|删除空行|分词|
|合并文件|删除中英文标点|词性标注|
|分割数据集|删除停用词|命名实体识别|
|-|删除乱码和特殊符号|依存句法分析|
|-|删除英文字符|语义角色标注|
* 目前没有向量化等功能，会不断完善

## 示例
* preprocess.py使用示例如下，输出结果均在注释中写出.
```python
from preprocess import Preprocess


test_equal1 = "句子1"
test_equal2 = "句子2"
print(Preprocess.is_equal(test_equal1, test_equal2))  # False

test_blank = ['句子1', '', '', '', '\n', '\t', '\r', '\f', '句子2']
print(Preprocess.del_blank_lines(test_blank))  # ['句子1', '句子2']

test_punc = ",标点符号*.s#?.<"
print(Preprocess.del_punctuation(test_punc))  # 标点符号s

test_seg_sents = [['今天', '天气', '真', '不错', '啊'], ['Tom', 'and', 'the', 'cat']]
test_stopwords = ['啊', '着', 'the', '真']
print(Preprocess.del_stopwords(test_seg_sents, test_stopwords))
# [['今天', '天气', '不错'], ['Tom', 'and', 'cat']]

test_length = "你好"
print(Preprocess.is_length_valid(test_length))  # False

test_simple_sent = "这不是一个简单句。真的不是。"
print(Preprocess.is_simple_sentence(test_simple_sent))  # False

test_special_symbol = "一些*&=-！特殊↓℃符号%"
print(Preprocess.del_special_symbol(test_special_symbol))  # 一些特殊符号

test_en_word = "需要nobody删除p20pro英文come符so0n号"
print(Preprocess.del_english_word(test_en_word))  # 需要删除20英文符0号

proc = Preprocess('./ltp_data_v3.4.0/')

test_ltp = ["小明，把电视安好，你若安好便是晴天，你若安不好...", "那咱家就可以换电视了"]
seged = proc.seg_sentences(test_ltp)
print(seged)
# [['小明', '，', '把', '电视', '安好', '，', '你', '若', '安好', '便是', '晴天', '，', '你', '若', '安', '不好', '...'],
# ['那', '咱', '家', '就', '可以', '换', '电视', '了']]

posed = proc.postag_sentences(seged)
print(posed)
# [['nh', 'wp', 'p', 'n', 'v', 'wp', 'r', 'v', 'a', 'v', 'n', 'wp', 'r', 'v', 'v', 'a', 'wp'],
# ['r', 'r', 'n', 'd', 'v', 'v', 'n', 'u']]

ne = proc.rec_named_entity(seged, posed)
print(ne)
# [['S-Nh', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

ne_sent = ['S-Nh', 'B-Ns', 'O', 'I-Ni', 'O', 'E-Nh']
print(Preprocess.get_ne_index(ne_sent))
# [0, 1, 3, 5]

arc_objs, arcs = proc.parse_dependency(seged, posed)
print(arcs)
# [[(5, 'SBV'), (1, 'WP'), (5, 'ADV'), (3, 'POB'), (0, 'HED'), (5, 'WP'), (9, 'SBV'), (9, 'ADV'), (5, 'COO'), (9, 'COO'), (10, 'VOB'), (10, 'WP'), (15, 'SBV'), (15, 'ADV'), (10, 'COO'), (15, 'CMP'), (5, 'WP')],
# [(3, 'ATT'), (3, 'ATT'), (6, 'SBV'), (5, 'ADV'), (6, 'ADV'), (0, 'HED'), (6, 'VOB'), (6, 'RAD')]]

roles = proc.label_sementic_role(seged, posed, arc_objs)
print(roles)
# [[(9, 'A1', 10, 10), (14, 'A0', 12, 12)],
# [(5, 'A2', 0, 2), (5, 'ADV', 3, 3), (5, 'A2', 6, 6)]]
```