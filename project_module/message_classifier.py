import json
import numpy as np
from pythainlp.word_vector import WordVector
from joblib import load

class MessageClassifier:
    def __init__(self, language='th'):
        self.model = None
        self.w2v_model = WordVector()
        self.classes = None
        self.language = language
    
    def load_model(self, model_path, json_classes_path):
        self.model = load(model_path) 
        self.__init_json_classes(json_classes_path)

    def classify(self, text_input = ''):
        text_vec = self.w2v_model.sentence_vectorizer(text_input)

        predict_proba = (self.model.predict_proba(text_vec))[0]
        max_confident = max(predict_proba)
        output = [ p for p in predict_proba ].index(max_confident)
        result = self.classes[output]

        return [result, max_confident]

    def predict(self, X):
        predict_proba_list = (self.model.predict_proba(X))
        max_confident_list = [ max(p) for p in predict_proba_list ]
        output = []
        for i in range(len(predict_proba_list)):
            result = [ p for p in predict_proba_list[i] ].index(max_confident_list[i])

            output.append(result)

        return output

    def predict_proba(self, X):
        return   self.model.predict_proba(X)

    def predict_sd(self, X):
        predict_proba = self.model.predict_proba(X)

        return [ np.std(p) for p in predict_proba ]

    def __init_json_classes(self, json_classes_path):
        with open(json_classes_path, 'r') as openfile:
            # Reading from json file
            temp_json = json.load(openfile)

        temp_json.sort(key=lambda item : item['category_target']) # sort classes by field cateogory target
        self.classes = [ item['category'] for item in temp_json ] 


    
