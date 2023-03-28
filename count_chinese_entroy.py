import re
from collections import Counter
import jieba
import math
from util import Read_file_list,combine2gram,combine3gram

#文本预处理
path_list = Read_file_list(r".\txt")
corpus = []
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        corpus += text

#去掉停词
with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
lines = [line.strip('\n') for line in lines]
for j in range(len(corpus)):
    for line in lines:
        corpus[j]=corpus[j].replace(line, "")
        corpus[j] = corpus[j].replace(" ", "")
regex_str = ".*?([^\u4E00-\u9FA5]).*?"
english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】〖〗《》？“”‘’！[\\]^_`{|}~]+'
symbol = []
for j in range(len(corpus)):
    corpus[j] = re.sub(english, "", corpus[j])
    symbol += re.findall(regex_str, corpus[j])
count_ = Counter(symbol)
count_symbol = count_.most_common()
noise_symbol = []
for eve_tuple in count_symbol:
    if eve_tuple[1] < 200:
        noise_symbol.append(eve_tuple[0])
noise_number = 0
for line in corpus:
    for noise in noise_symbol:
        line=line.replace(noise, "")
        noise_number += 1
print("完成的噪声数据替换点：", noise_number)

#计算1-Gram中文信息熵
token = []
for para in corpus:
    token += jieba.lcut(para)
token_num = len(token)
ct = Counter(token)
vocab1 = ct.most_common()
entropy_1gram = sum([-(eve[1]/token_num)*math.log((eve[1]/token_num),2) for eve in vocab1])
print("1-Gram 词库总词数：", token_num, " ", "不同词的个数：", len(vocab1))
print("出现频率前10的1-gram词语：", vocab1[:10])
print("entropy_1gram:", entropy_1gram)

#计算2-Gram中文信息熵
token_2gram = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_2gram += combine2gram(cutword_list)
token_2gram_num = len(token_2gram)
ct2 = Counter(token_2gram)
vocab2 = ct2.most_common()
same_1st_word = [eve.split(" ")[0] for eve in token_2gram]
assert token_2gram_num == len(same_1st_word)
ct_1st = Counter(same_1st_word)
vocab_1st = dict(ct_1st.most_common())
entropy_2gram = 0

for eve in vocab2:
    p_xy = eve[1]/token_2gram_num
    first_word = eve[0].split(" ")[0]
    entropy_2gram += -p_xy*math.log(eve[1]/vocab_1st[first_word], 2)
print("2-Gram 词库总词数：", token_2gram_num, " ", "不同词的个数：", len(vocab2))
print("出现频率前10的2-gram词语：", vocab2[:10])
print("entropy_2gram:", entropy_2gram)

###计算3-Gram中文信息熵
token_3gram = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_3gram += combine3gram(cutword_list)
token_3gram_num = len(token_3gram)
ct3 = Counter(token_3gram)
vocab3 = ct3.most_common()
same_2st_word = [eve.split(" ")[0] for eve in token_3gram]
assert token_3gram_num == len(same_2st_word)
ct_2st = Counter(same_2st_word)
vocab_2st = dict(ct_2st.most_common())
entropy_3gram = 0
for eve in vocab3:
    p_xyz = eve[1]/token_3gram_num
    first_2word = eve[0].split(" ")[0]
    entropy_3gram += -p_xyz*math.log(eve[1]/vocab_2st[first_2word], 2)
print("3-Gram词库总词数：", token_3gram_num, " ", "不同词的个数：", len(vocab3))
print("出现频率前10的3-gram词语：", vocab3[:10])
print("entropy_3gram:", entropy_3gram)
