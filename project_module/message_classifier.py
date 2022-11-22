import json
import numpy as np
from pythainlp.word_vector import WordVector
from joblib import load

class MessageClassifier:
    def __init__(self, error_sd= 0.07765331184188104, language='th'):
        self.model = None
        self.w2v_model = WordVector()
        self.classes = None
        self.error_sd = error_sd
        self.language = language
    
    def load_model(self, model_path, json_classes_path):
        self.model = load(model_path) 
        self.__init_json_classes(json_classes_path)

    def classify(self, text_input = ''):
        text_vec = self.w2v_model.sentence_vectorizer(text_input)
        output = self.model.predict(text_vec)

        result = self.classes[output[0]]

        predict_sd = self.predict_sd(text_vec)
        print(predict_sd)

        max_confident = max((self.model.predict_proba(text_vec))[0])
        # if(max_confident < self.common_offset):
        #     return [f'{result}->C', (-1 * max_confident)]

        return [result, max_confident]

    def predict(self, X):
        output = self.model.predict(X)

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


    
