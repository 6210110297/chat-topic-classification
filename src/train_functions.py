import time
import pandas as pd
import numpy as np
import seaborn as sns
import re
import numpy as np
import matplotlib.pyplot as plt
import json

from wordcloud import WordCloud
from pythainlp.tokenize import THAI2FIT_TOKENIZER  # ใช้ในการตัดคำ

def clean_data(data):
    # sort
    data = data.sort_values(by=['category'])

    # reset index and filter column
    data = data.reset_index()
    data = data[['message', 'category']]

    return data


def plot_count_graph(data):
    count_graph = sns.countplot(data)
    count_list = []

    for p in count_graph.patches:
        height = p.get_height()
        count_list.append(height)
        count_graph.annotate('{:.1f}'.format(
            height), (p.get_x()+0.25, height+0.01))

    plt.title('Data Each Category')
    plt.show()


def balance_data(nb_data, field, field_list, n=100):
    # temp list
    df_list = []

    for f in field_list:
        f = f"'{f}'"  # stringify

        # query data by data per category
        temp_data = nb_data.query(f"{field} == {f}").sample(
            n=n,
            replace=False,  # True if numbers of sample higher than minimum numbers of category
            random_state=18,
        )

        df_list.append(temp_data)

    return df_list

def count_words(data, category=''):
    word_list = set()
    if(category != ''):
        sub_frame = data[data['category']==category]
    else:
        sub_frame = data

    for text in sub_frame['message']:
        
        text = text.lower().replace('\n', ' ').replace('\r', '').strip()
        text = re.findall(r"[\u0E00-\u0E7Fa-zA-Z']+", text)
        text = ' '.join(text)

        word_tokens = THAI2FIT_TOKENIZER.word_tokenize(text)
        filtered_sentence = set([w for w in word_tokens])

        word_list.update(filtered_sentence)

    return len(word_list)

def create_wordcloud(words, title):
    wordcloud = WordCloud(
        font_path='THSarabun.ttf', # path ที่ตั้ง Font
        regexp=r"[\u0E00-\u0E7Fa-zA-Z']+" # ป้องกัน bug วรรณยุกต์
    ).generate(' '.join(THAI2FIT_TOKENIZER.word_tokenize(words)))

    plt.figure(figsize=[10, 7])
    plt.imshow(wordcloud, interpolation= "bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()
    

def check_wordcloud(data, category):
    subset = data[data.category==category]
    text = subset.message.values
    words = ''.join(text)
    create_wordcloud(words, category)
