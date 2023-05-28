
import pandas as pd
import re
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec



# Sayısal değerlerin kaldırılması

def remove_numeric(value):
    bfr=[item for item in value if not item.isdigit()]
    bfr="".join(bfr)
    return bfr



# Emojilerin kaldırılması
def remove_emoji(value):
    bfr=re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # simgeler ve pictographs
                           u"\U0001F680-\U0001F6FF"  # taşıtlar ve semboller
                           u"\U0001F1E0-\U0001F1FF"  # bayraklar (iOS)
                           u"\U00002702-\U000027B0"  # diğer emojiler
                           u"\U000024C2-\U0001F251"  # diğer emojiler
                           "]+",flags=re.UNICODE)
    bfr=bfr.sub(r'',value)
    return bfr



# tek karakterli ifadelerin kaldırılması
def remove_single_chracter(value):
    return re.sub(r'\b\w\b', '', value)



# noktalama işaretlerinin kaldırılması
def remove_noktalama(value):
    return re.sub(r'[^\w\b]','',value)



# linklerin kaldırılması
def remove_link(value):
    return re.sub(r'http\S+|www\S+', '', value)


# hashtaglerin kaldırılması
def remove_hashtag(value):
    return re.sub(r'#\w+', '', value)


# username in kaldırılması
def remove_username(value):
    return re.sub(r'@\w+', '', value)


# kök indirgeme ve stop words işlemleri
def stem_word(value):
    stemmer=snowballstemmer.stemmer("turkish")
    value=value.lower()
    value=stemmer.stemWords(value.split())
    stop_words=['acaba','ama','aslında','az','bazı','belki','biri','çok',
                'çünkü','da','de','defa','eğer','en','gibi','hem','siz','şu']
    value=[item for item in value if not item in stop_words]
    value=' '.join(value)
    return value


def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_chracter
                           (remove_noktalama
                            (remove_link
                             (remove_hashtag
                              (remove_username
                               (stem_word
                                (word)))))))) for word in value.split()]


# boşlukların kaldırılması
def remove_space(value):
    return [item for item in value if item.strip()]


# bag of words model

def bag_of_words(value):
    vectorizer = CountVectorizer()
    X=vectorizer.fit_transform(value)
    return X.toarray().tolist()


# tf-idf model

def tfidf(value):
    vectorizer=TfidfVectorizer()
    X=vectorizer.fit_transform(value)
    return X.toarray().tolist()


# word2vec model

def word2vec(value):
    model=Word2Vec.load("data/word2vec.model")
    bfr_list=[]
    bfr_len=len(value)
    
    for k in value:
        bfr=model.wv.key_to_index[k]
        bfr=model.wv[bfr]
        bfr_list.append(bfr)

    bfr_list=sum(bfr_list)
    bfr_list=bfr_list/bfr_len
    return bfr_list.tolist()


