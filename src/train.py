import time
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import json


# preprocessing
from sklearn import preprocessing

# Word2Vec
from pythainlp.word_vector import WordVector

# AI
from sklearn.model_selection import train_test_split  # split data set

# report train & test result
from sklearn.metrics import accuracy_score, classification_report

# AI Models
from sklearn.neural_network import MLPClassifier

# Saving Intelligence
from joblib import dump

# import all required train functions
from train_functions import *

def train():
    with open('./src/config.json', 'r') as openfile:
            # Reading from json file
            config = json.load(openfile)

    # env variables
    visulize = config['VISULIZE']
    data_path = config['DATA_PATH']
    data_per_category = config['DATA_PER_CATEGORY']
    class_path = config['CLASS_PATH']
    dst_dir = config['DST_DIR']

    # read and select data
    data = pd.read_csv(data_path)
    data = data[['message', 'category']]

    # check data head
    if (visulize):
        print(data.head())

    # clean data
    data = clean_data(data)

    # keep category list
    category_list = data['category'].unique()

    # check data properties
    if (visulize):
        print(f'data categories: {category_list}')
        print('----------------------------------------------------------------')
        print(f'shape: {data.shape}')
        print(f'types: {data.dtypes}')
        print(f'isNull: {data.isnull().any()}')

        # plot count graph
        plot_count_graph(data['category'])
       

    # balancing data
    temp_list = balance_data(
        data,
        field = 'category',
        field_list = category_list,
        n = data_per_category
    )
    # create new data frame from temp list
    data = clean_data(data= pd.concat(temp_list))

    # check data properties
    if (visulize):
        # plot count graph
        plot_count_graph(data['category'])
       

        # plot message length
        data['message_length'] = data['message'].str.len()
        # sns.displot(data['message_length']).set_titles('message_length')
       

        # word count data
        words_category = []
        for category in category_list:
            words_category.append(count_words(data, category))
        # word count graph
        words_count = {'category': category_list, 'words_count': words_category}
        w_count_data = pd.DataFrame(words_count)
        # display
        print(f'total words: {count_words(data)}')
        # sns.barplot(x= 'category', y='words_count', data=w_count_data)
       

        # word cloud check
        cat_string = 'QSTA' #character list of category
        for c in cat_string:
            check_wordcloud(data, c)
           

    label_encoder = preprocessing.LabelEncoder()
    data['category_target'] = label_encoder.fit_transform(data['category'])
  
    # keep category target in json
    temp_data = data[['category', 'category_target']].drop_duplicates()
    json_classes = temp_data.to_json(orient="records")
    json_file = class_path
    # Writing to .json
    with open(json_file, "w") as outfile:
        outfile.write(json_classes)

    #  Word to Vector
    w2v_thai = WordVector()
    word2vec = [
        w2v_thai.sentence_vectorizer(
            data['message'][i]
        ) 
    for i in range(0, len(data['message']))
    ]
    X = np.array(word2vec).tolist()
    data['word2vec'] = X

    if(visulize):
        print('----------------------------------------------------------------')
        print(data.head())

    # prepare train data
    data_temp = [ x for x in data['word2vec']]
    data_reshape = np.reshape(data_temp, (-1, 300))
    # split test and train
    X_train, X_test, y_train, y_test = train_test_split(
        data_reshape,
        data['category_target'],
        random_state=18,
        test_size = 0.2,
        shuffle=True
    )

    # train
    # MLP Model
    model = MLPClassifier(
        hidden_layer_sizes= (20,), 
        random_state= 18
    )
    model.fit(X_train, y_train)
    
    # accuracy result
    accuracy_predictions = model.predict(X_train)
    print('Validation', accuracy_score(y_train, accuracy_predictions))

    model_predictions = model.predict(X_test)

    print('Accuracy', accuracy_score(y_test, model_predictions))

    print(classification_report(y_test, model_predictions))

    # save model
    time_stamp = int(time.time())
    model_name = f'{dst_dir}/model{time_stamp}.joblib'
    print(f'model saved : {model_name}')
    dump(model, f'{dst_dir}/model{time_stamp}.joblib')

if __name__ == '__main__':
    train()
