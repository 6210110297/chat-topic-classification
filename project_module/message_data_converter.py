import re
import pandas as pd
import json

#file management
from os import listdir
from os.path import isfile, join

from .message_process import MessageProcess

class MessageDataConverter:
    def __init__(self):
        self.message_list = None
        self.message_process = MessageProcess()

    def convert_many(self, json_path, csv_path):
        file_list = [f for f in listdir(json_path) if isfile(join(json_path, f))]
        for f in file_list:
            file_name = f.split('.')[0]
            self.import_json(path = f'{json_path}{f}')
            self.export_csv(path = f'{csv_path}{file_name}.csv')
            self.clear_message_list()

    def import_json(self, path):
        json_file = open(path)
        json_obj = json.load(json_file)
        message_list = []

        for message in json_obj['messages']:
            try:
                message_content = message['content'].encode('latin_1').decode('utf-8')
                message_list.append(message_content)
                
            except:
                continue

        if self.message_list:
            self.message_list.extend(message_list)
        else:
            self.message_list = message_list

        # filter message
        self.message_list = self.message_process.filter_none_thai_character(message_list)

    def export_csv(self, path):
        data = pd.DataFrame({'message': self.message_list})

        # process thai text
        data['message_parsed'] = data['message'].apply(self.message_process.process_thai_text)

        # assign default category
        data['category'] = 'C'
        
        data.to_csv(path, encoding='utf-8-sig')

    def merge_csv(self, csv1_path, csv2_path, des_path):
        csv1 = pd.read_csv(csv1_path)
        csv2 = pd.read_csv(csv2_path)
        merged = pd.concat([csv1, csv2], ignore_index= True)
        
        merged.to_csv(des_path, encoding='utf-8-sig')

    def clear_message_list(self):
        self.message_list = None


   
