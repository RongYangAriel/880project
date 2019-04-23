import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk import word_tokenize
import string


class TweetProcess(object):

    def __init__(self, files):
        self.path = files

    def list_files(self):
        entire_list = []
        pd_data = pd.DataFrame()
        for p in path_dates:
            entire_list.append(self.path.format(p))
        # print(entire_list)
        for e in entire_list:
            df = pd.read_json(e, lines=True)
            pd_data = pd_data.append(df)
        return pd_data
        # pd_data['created_at'].dt.date.to_csv('/Users/rohan/Documents/Grad School/'
        #                                      'Coursework/Winter/ML & NLP/list_files.txt', header='created_at')
        # return pd_data['created_at'].dt.date.reset_index(drop=True)

    def give_list_new(self):
        empty_df = pd.DataFrame()
        created_at = pd.DataFrame()
        # print(self.list_files()['text'])
        empty_df['text'] = (self.list_files()['text'])
        created_at['Date'] = self.list_files()['created_at'].dt.date.reset_index(drop=True)

        empty_df.to_csv('/Users/rohan/Documents/Grad School/coursework/winter/ML & NLP/project_code/data/give_list.csv')
        created_at.to_csv('/Users/rohan/Documents/Grad School/coursework/winter/ML & NLP/project_code/data'
                          '/created_at.csv')


    def preprocess_new(self):
        # string_df_words = self.give_list_new()
        string_df_words = pd.read_csv('/Users/rohan/Documents/Grad School/coursework/winter/ML & NLP/project_code/data'
                                      '/give_list.csv')
        string_df_words.loc[:, "text"] = string_df_words.text.apply(lambda x: str.lower(x))
        string_df_words.loc[:, "text"] = string_df_words.text.apply(lambda x: " ".join(re.findall('[\w]+', x)))
        # translator = str.maketrans('', '', string.digits)
        # string_df_words.loc[:, "text"] = string_df_words.text.apply(lambda x: x.translate(translator))
        string_df_words.loc[:, "text"] = string_df_words.text.apply(lambda x: word_tokenize(x))
        # string_df_words.loc[:, "text"] = string_df_words.apply(lambda x: x.replace('[^a-zA-Z]', ''))
        new_words = ['url', 'rt', 'at_user']
        stop = set(stopwords.words('english')).union(new_words)
        string_df_words.loc[:, "text"] = string_df_words.text.apply((lambda x:
                                                                     [item for item in x if item not in stop]))
        string_df_words.loc[:, "text"] = string_df_words.text.apply(lambda x: str.join(' ', x))
        return string_df_words

    def final_file(self):
        new_df = self.preprocess_new()
        # new_df = pd.read_csv('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/preprocessed.csv')
        new_df = new_df.drop(['Unnamed: 0'], axis=1)
        return new_df

    def group_func(self):
        temp_df = pd.read_csv('/Users/rohan/Documents/Grad School/Coursework/winter/ML & NLP/Stock_Project/Our_Bert/'
                              'Bert_Data/created_at.csv')\
            .drop(['Unnamed: 0'], axis=1)
        df = self.final_file()
        temp_df['text'] = df['text']
        # df = df.groupby('created_at')['text'].apply(','.join).reset_index()
        # max_list = []
        # for i in range(0, 695):
        #     max_list.append(len(df['text'][i].split()))
        # return max(max_list)
        return temp_df
        df.to_csv('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/new_dataset.csv')

    def create_signals(self):
        address = '/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/' \
                  'Stock_Project/stocknet-dataset/price/raw/' + COMPANY_NAME + '.csv'
        # print(address)
        df = pd.read_csv(address)
        # print(df)
        diff = []
        status = []
        signal = []
        for i in range(len(df['Date']) - 1):
            diff.append(df.loc[i + 1]['Open'] - df.loc[i]['Open'])
            status.append(diff[i] / df.loc[i]['Open'])
            if status[i] > 0.01:
                signal.append('up')
            elif status[i] < -0.01:
                signal.append('down')
            else:
                signal.append('stay')
        print(df)
        diff.append(1)
        status.append(0)
        signal.append('up')
        # print(len(diff))
        # print(len(signal))
        # df.insert(7,'diff',diff)
        # df.insert(8,'increase precent',status)
        df.insert(7, 'signal', signal)
        # print(df)
        df = df.loc[0:1256]
        df = df.set_index(['Date'])
        df = df.loc[str(min(self.list_files()['created_at'].dt.date.reset_index(drop=True))):str(max(self.list_files()['created_at'].dt.date.reset_index(drop=True)))]
        pd.DataFrame(df['signal']).to_csv('/Users/rohan/Documents/Grad School/Coursework/winter/ML & NLP/'
                                          'nyt_data/final_files' + COMPANY_NAME + '.csv')

    def combine_with_signals(self):
        signals_df = pd.read_csv('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/nyt_data/'
                                 'ticker_signal/' + COMPANY_NAME + '.csv')
        # print(type(self.group_func()), type(signals_df))
        # print(new_df)
        df = self.group_func().groupby('Date')['text'].apply(','.join).reset_index()
        df.Date = pd.to_datetime(df.Date)
        signals_df.Date = pd.to_datetime(signals_df.Date)
        new_df = pd.merge_asof(df, signals_df, on='Date')
        for i in range(0, len(new_df)):
            new_df['Ticker'] = COMPANY_NAME
        print(new_df)
        new_df.to_csv('/Users/rohan/Documents/Grad School/Coursework/winter/ML & NLP/nyt_data/final_files/'
                      + COMPANY_NAME + '.txt')


company_list = pd.read_excel('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/company-name.xlsx')
for i in company_list['Symbol']:
    COMPANY_NAME = i

    path = '/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/Stock_Project/stocknet-dataset/tweet/' \
           'preprocessed/' + COMPANY_NAME
    path_dates = os.listdir(path)

    path = f'/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/Stock_Project/stocknet-dataset/tweet/' \
           'preprocessed/' + COMPANY_NAME + '/{}'

    tweets = TweetProcess(path)
    tweets.give_list_new()
    # tweets.create_signals()
    tweets.combine_with_signals()
