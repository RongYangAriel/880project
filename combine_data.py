import pandas as pd

company_list = pd.read_excel('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/company-name.xlsx')
security_stocks = pd.read_csv('/Users/rohan/Documents/Grad School/Coursework/winter/ML & NLP/Stock_Project'
                              '/Our_Bert/texts_and_fin2.csv')
df = pd.DataFrame()
df2 = pd.DataFrame()
df2['signal'] = security_stocks['signal']
df2['Date'] = security_stocks['release_date']
df2['text'] = security_stocks['filtered_text3']
df2['Ticker'] = security_stocks['ticker']
# df2 = df2.drop(['Unnamed: 0'], axis = 1)

for i in company_list['Symbol']:
    path = '/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/Bert_Data/' + i + '.txt'
    df = df.append(pd.read_csv(path)).drop(labels='Unnamed: 0', axis=1)
# df2['Date'] = pd.to_datetime(df2['Date']).dt.date
df = df.append(df2)
df['Date'] = pd.to_datetime(df['Date']).dt.date
# df.drop(['Unnamed: 0.1', 'Ticker'])
df.to_csv('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/nyt_data/bert_pass_123.txt')
print(df.shape)
# print(df.head(), df.tail())
# df = pd.read_csv('/Users/rohan/Documents/Grad School/Coursework/Winter/ML & NLP/Bert_Data/BIG_DATA_8K.txt')
