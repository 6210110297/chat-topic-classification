import re
import json
from pythainlp.tokenize import word_tokenize # ใช้ในการตัดคำ
from pythainlp.corpus import common # ใช้ลบคำที่ไม่ใช้ออก
from pythainlp.word_vector import WordVector
from joblib import dump, load

class MessageClassifier:
    def __init__(self, language='th'):
        self.model = None
        self.w2v_model = WordVector()
        self.classes = None
        self.language = language
    
    def load_model(self, model_path, json_classes_path):
        self.model = load(model_path) 
        # self.w2v_model = load(w2v_path)
        self.__init_json_classes(json_classes_path)

    def classify(self, text_input = ''):
        text_vec = self.w2v_model.sentence_vectorizer(text_input)
        output = self.model.predict(text_vec)

        return self.classes[output[0]]

    def __init_json_classes(self, json_classes_path):
        with open(json_classes_path, 'r') as openfile:
            # Reading from json file
            temp_json = json.load(openfile)

        temp_json.sort(key=lambda item : item['category_target']) # sort classes by field cateogory target
        self.classes = [ item['category'] for item in temp_json ] 

    def __process_thai_text(self, text):
        text = text.lower().replace('\n', ' ').replace('\r', '').strip()
        text = re.findall(r"[\u0E00-\u0E7Fa-zA-Z']+", text)
        text = ''.join(text)

        stop_words = common.thai_stopwords()
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        text = ' '.join(filtered_sentence)
        return text

    
