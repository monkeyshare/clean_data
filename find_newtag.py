from itertools import combinations
import re
import nltk
from collections import Counter
import pandas as pd
import itertools
import pymysql
from redis import StrictRedis
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
user_stops = []

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!')

def iter_tool(title, n):
    bigram_vectorizer = CountVectorizer(ngram_range=(1,n),token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    w_list=analyze(title)
    return w_list


def get_t():
    """
    导入已经合并了变体的类目数据，返回t
    """
    t = pd.read_csv('top_all.csv', engine='python')
    return list(t['Title'])


def get_clean_title(t):
    '''
    返回类目下单词及词频统计结果result_word、清洗好的标题result_title
    '''
    word_dics = {}
    title_lis = []
    for title in t:
        title = get_raw_title(title)  # 清洗标题
        title_lis.append(' '.join(title))
    return title_lis


def get_result_dics(t_raw_list,n):
    dics = {}
    title_id=0
    for title in t_raw_list:
        title_id+=1
        w_lis=iter_tool(title,n=n)
        print(title_id, len(w_lis))  # 打印第*个标题及该标题拆出来的关键词的数量
        for key_words in w_lis:
            if key_words not in dics:
                dics[key_words] = 1
            else:
                dics[key_words]+= 1
    lis=[]
    for i,n in dics.items():
        lis.append([i,n])
    return lis

def main(n):
    t_clean_list = get_t()
    print(len(t_clean_list))
    result_dics = get_result_dics(t_clean_list,n=n)
    return result_dics

if __name__ == '__main__':
    result_lis = main(n=3)
    print(len(result_lis))
    t=pd.DataFrame(result_lis)
    t['lw']=t[0].apply(lambda x:len(x.split()))
    t.columns=['word','freq','lw']
    t.to_csv('111.csv')
    




