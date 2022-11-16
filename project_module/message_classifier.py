import re
import json
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
        output = self.model.predict(text_vec)

        result = self.classes[output[0]]
        max_confident = max((self.model.predict_proba(text_vec))[0])
        if(max_confident < 0.5):
            return ['C', -1]

        return [result, max_confident]

    def __init_json_classes(self, json_classes_path):
        with open(json_classes_path, 'r') as openfile:
            # Reading from json file
            temp_json = json.load(openfile)

        temp_json.sort(key=lambda item : item['category_target']) # sort classes by field cateogory target
        self.classes = [ item['category'] for item in temp_json ] 


    
