import pandas as pd
import json

#file management
from os import listdir
from os.path import isfile, join

#time stamp
import time

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

    def merge_many_csv(self, src_path, des_path):
        time_stamp = int(time.time())
        file_list = [f for f in listdir(src_path) if isfile(join(src_path, f))]
        i = 0
        for f in file_list:
            file_path = f'{src_path}{f}'
            if(i==0):
                merged = pd.read_csv(file_path)
                i+=1
                continue

            csv1 = merged
            csv2 = pd.read_csv(file_path)

            merged = pd.concat([csv1, csv2], ignore_index= True)
            
        merged = self.__clean_data__(merged)
        merged.to_csv(f'{des_path}data{time_stamp}.csv', encoding='utf-8-sig')


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

        # assign default category
        data['category'] = 'C'
        
        data.to_csv(path, encoding='utf-8-sig')

    def clean_csv(self, path):
        data = pd.read_csv(path)
        data = self.__clean_data__(data)

        data.to_csv(path, encoding='utf-8-sig')

    def clear_message_list(self):
        self.message_list = None

    def print_message_list(self):
        return print(self.message_list)

    def __clean_data__(self, data):
        #filter na
        data = data.dropna()
        #sort by category
        data = data.sort_values(by=['category'])
        #reset index
        data = data.reset_index()
        data = data[['message', 'category']]

        return data


   
