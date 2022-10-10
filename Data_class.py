import gensim
import json
import pandas as pd
import numpy as np

class Data_class(object):
    """класс для получения данных из файлов"""
    def GetData(jsonl): #jsonl → данные
        #загрузка jsonl
        messages = pd.read_json(path_or_buf=jsonl, lines=True)
        #названия столбцов
        messages.columns = ["text", "label"]
        messages.head()
        #очищение данных с помощью gensim
        messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
        messages.head()

        x_train = messages['text_clean']
        y_train = messages['label']
        return x_train, y_train

