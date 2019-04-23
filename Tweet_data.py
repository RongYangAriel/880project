import os
import pandas as pd
import numpy as np

if __name__ =='__main__':
    address = '/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/' \
                    'Stock_Project/stocknet-dataset/price/raw/AAPL.csv'
    # print(address)
    df = pd.read_csv(address)
    # print(df)
    diff = []
    status =[]
    signal = []
    for i in range(len(df['Date'])-1):
        diff.append(df.loc[i+1]['Open'] - df.loc[i]['Open'])
        status.append(diff[i]/ df.loc[i]['Open'])
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
    df.insert(7,'signal',signal)
    # print(df)
    df = df.loc[0:1256]
    df = df.set_index(['Date'])
    df = df.loc['2013-12-31':'2015-12-31']
    pd.DataFrame(df['signal']).to_csv('/Users/rohan/Documents/Grad School/Coursework/winter/ML & NLP/'
                                      'Bert_Data/update_signals.csv')

    # df.to_csv('/Users/rohan/Documents/Grad School/Coursework/winter/ML & NLP/signals.csv')

