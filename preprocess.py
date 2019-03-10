"""本文件包含中文NLP预处理常用的一些代码"""

import os
import re
import string
import numpy as np
from pyltp import *
from zhon.hanzi import punctuation


class Preprocess(object):
    """中文NLP预处理类"""

    # 用来处理数据的正则表达式
    DIGIT_RE = re.compile(r'\d+')
    LETTER_RE = re.compile(r'[a-zA-Z]+')
    SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 用以删除一些特殊符号
    NAMED_ENTITY = re.compile(r'[SBIE]+')
    STOPS = ['。', '.', '?', '？', '!', '！']  # 中英文句末字符

    # 句子所限制的最小,最大长度
    SENTENCE_MIN_LEN = 5
    SENTENCE_MAX_LEN = 50

    def __init__(self, ltp_model_dir):
        self._cws_model_path = os.path.join(ltp_model_dir, 'cws.model')
        self._pos_model_path = os.path.join(ltp_model_dir, 'pos.model')
        self._ner_model_path = os.path.join(ltp_model_dir, 'ner.model')
        self._par_model_path = os.path.join(ltp_model_dir, 'parser.model')
        self._srl_model_path = os.path.join(ltp_model_dir, 'pisrl_win.model')

    @staticmethod
    def read_text_file(text_file):
        """读取文本文件,并返回由每行文本作为元素组成的list."""
        with open(text_file, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
        return lines

    @staticmethod
    def write_text_file(text_list, target_file):
        """将文本列表写入目标文件

        Args:
            text_list: 列表，每个元素是一条文本
            target_file: 字符串，写入目标文件路径
        """
        with open(target_file, 'w', encoding='utf-8') as writer:
            for text in text_list:
                writer.write(text + '\n')

    @staticmethod
    def merge_files(filedir, target_file):
        """
        合并一个文件夹中的文本文件。注意：需要合并的每个文件的结尾要有换行符。

        Args:
            filedir: 需要合并文件的文件夹
            target_file: 合并后的写入的目标文件
        """
        filenames = os.listdir(filedir)
        with open(target_file, 'a', encoding='utf-8') as f:
            for filename in filenames:
                filepath = os.path.join(filedir, filename)
                f.writelines(open(filepath, encoding='utf-8').readlines())

    @staticmethod
    def partition_dataset(dataset, ratio):
        """将一个大的数据集按比例切分为训练集、开发集、测试集

        Args:
            dataset: 列表，原始数据集
            ratio: 三元组，训练集、开发集、测试集切分比例，每个元素为0-1之间的小数

        Returns: train, val, test, 表示训练集、开发集、测试集的三个列表
        """
        data_len = len(dataset)
        train_len = int(np.floor(data_len * ratio[0]))
        val_len = int(np.ceil(data_len * ratio[1]))
        test_len = data_len - train_len - val_len
        return dataset[:train_len], dataset[train_len: -test_len], dataset[-test_len:]

    @staticmethod
    def is_equal(sent1, sent2):
        """判断两个句子是否完全相同"""
        return sent1 == sent2

    @staticmethod
    def del_blank_lines(sentences):
        """删除句子列表中的空行，返回没有空行的句子列表

        Args:
            sentences: 字符串列表
        """
        return [s for s in sentences if s.split()]

    @staticmethod
    def del_punctuation(sentence):
        """删除字符串中的中英文标点.

        Args:
            sentence: 字符串
        """
        en_punc_tab = str.maketrans('', '', string.punctuation)  # ↓ ① ℃处理不了
        sent_no_en_punc = sentence.translate(en_punc_tab)
        return re.sub(r'[%s]+' % punctuation, "", sent_no_en_punc)

    @staticmethod
    def del_stopwords(seg_sents, stopwords):
        """删除句子中的停用词

        Args:
            seg_sents: 嵌套列表，分好词的句子（列表）的列表
            stopwords: 停用词列表

        Returns: 去除了停用词的句子的列表
        """
        return [[word for word in sent if word not in stopwords]for sent in seg_sents]

    @classmethod
    def is_length_valid(cls, sentence):
        """限制句子长度,判断是否合法."""
        return cls.SENTENCE_MIN_LEN <= len(sentence) <= cls.SENTENCE_MAX_LEN

    @classmethod
    def is_simple_sentence(cls, sentence):
        """判断是否是简单句。
        简单句在这里定义为句子中只有一个句末终止符的句子，这样的句子含义比较明确。"""
        counter = 0
        for word in sentence:
            if word in cls.STOPS:
                counter += 1
                if counter > 1:
                    return False
        return True

    @classmethod
    def del_special_symbol(cls, sentence):
        """删除句子中的乱码和一些特殊符号。"""
        return cls.SPECIAL_SYMBOL_RE.sub('', sentence)

    @classmethod
    def del_english_word(cls, sentence):
        """删除句子中的英文字符"""
        return cls.LETTER_RE.sub('', sentence)

    @classmethod
    def get_ne_index(cls, ne_sent):
        """获取命名实体在句子中的位置。

        Args:
            ne_sent: 命名实体标记构成的列表
        """
        return [idw for idw, word in enumerate(ne_sent) if cls.NAMED_ENTITY.match(word)]

    def seg_sentences(self, sentences):
        """对输入的字符串列表进行分词处理,返回分词后的字符串列表."""
        segmentor = Segmentor()
        segmentor.load(self._cws_model_path)
        seg_sents = [list(segmentor.segment(sent)) for sent in sentences]
        segmentor.release()
        return seg_sents

    def postag_sentences(self, seg_sents):
        """对分完词的句子列表进行词性标注,返回标注的词性列表

        Args:
            seg_sents: 分好词的语句列表,每个语句也是一个列表
        """
        postagger = Postagger()
        postagger.load(self._pos_model_path)
        pos_sents = [list(postagger.postag(sent)) for sent in seg_sents]
        postagger.release()
        return pos_sents

    def rec_named_entity(self, seg_sents, pos_sents):
        """命名实体识别

        Args:
            seg_sents: 分好词的语句列表,每个语句也是一个列表
            pos_sents: 词性标注好的词性列表,每个语句的词性也是一个列表

        Returns: 命名实体识别完的语句列表,每个语句的命名实体识别结果也是一个列表
        """
        recognizer = NamedEntityRecognizer()
        recognizer.load(self._ner_model_path)
        ne_sents = [list(recognizer.recognize(seg_sents[i], pos_sents[i])) for i in range(len(seg_sents))]
        recognizer.release()
        return ne_sents

    def parse_dependency(self, seg_sents, pos_sents):
        """依存句法分析

        Args:
            seg_sents: 分好词的语句列表,每个语句也是一个列表
            pos_sents: 词性标注好的词性列表,每个语句的词性也是一个列表

        Returns:
            arc_objs: pyltp.VectorOfParseResult对象，依存句法分析结果对象的列表。
            arc_sents: 依存句法分析完的语句列表,每个语句的依存句法分析结果也是一个列表
        """
        parser = Parser()
        parser.load(self._par_model_path)
        arc_objs = [parser.parse(seg_sents[i], pos_sents[i]) for i in range(len(seg_sents))]
        arc_sents = [[(a.head, a.relation) for a in arc] for arc in arc_objs]
        parser.release()
        return arc_objs, arc_sents

    def label_sementic_role(self, seg_sents, pos_sents, arc_sents):
        """语义角色标注

        Args:
            seg_sents: 分好词的语句列表，每个语句也是一个列表
            pos_sents: 词性标注好的词性列表，每个语句的词性也是一个列表
            arc_sents: 依存句法分析结果列表，每个语句的依存句法分析结果也是一个列表

        Returns: 语义角色标注完的语句列表，每个语句的语义角色标注结果也是一个列表
        """
        labeler = SementicRoleLabeller()
        labeler.load(self._srl_model_path)

        roles = [labeler.label(seg_sents[i], pos_sents[i], arc_sents[i]) for i in range(len(seg_sents))]

        _ret = []
        for role in roles:
            _role = []
            for r in role:
                _role.extend([(r.index, arg.name, arg.range.start, arg.range.end) for arg in r.arguments])
            _ret.append(_role)

        labeler.release()

        return _ret

