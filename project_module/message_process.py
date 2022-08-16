import re
from pythainlp.tokenize import word_tokenize # ใช้ในการตัดคำ
from pythainlp.corpus import common # ใช้ลบคำที่ไม่ใช้ออก

class MessageProcess:
    def __init__(self):
        self.thai_characters = self.__init_thai_characters()

    def filter_none_thai_character(self, message_list):
        filter_func = lambda s: any(x in s for x in self.thai_characters)
        message_list_filtered = [line for line in message_list if filter_func(line)]

        return message_list_filtered

    def process_thai_text(self, text):
        text = text.lower().replace('\n', ' ').replace('\r', '').strip()
        text = re.findall(r"[\u0E00-\u0E7Fa-zA-Z']+", text)
        text = ''.join(text)

        stop_words = common.thai_stopwords()
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        text = ' '.join(filtered_sentence)
        return text

    def __init_thai_characters(self):
        # use constanst values to create map
        thai_characters_offset = 14727297
        thai_characters_bytes_len = 3
        thai_characters = [ (index.to_bytes(thai_characters_bytes_len, 'big')).decode('utf-8') for index in range(thai_characters_offset, thai_characters_offset + 63) ]

        return thai_characters
