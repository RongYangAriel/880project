import pandas as pd
import numpy as np
import requests
import io

company_list = ['AAPL','AMZN','FB','GOOGL','MSFT']
for name in company_list:
    scrape_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + str(name) + '&outputsize=full&apikey=W5S4WZGZ683B5EJP&datatype=csv'
    price = requests.get(scrape_url).content
    df = pd.read_csv(io.StringIO(price.decode('utf-8')))
    # df['date'] = pd.date_range('2000-1-1', periods=200, freq='D')
    df = df.loc[:,'timestamp':'open']
    df['ticker'] = name
    mask = (df['timestamp'] >= '2014-12-31') & (df['timestamp'] <= '2018-12-31')
    print(df.loc[mask])
    df = df.loc[mask]
    df.index = range(len(df['timestamp']))
    # print(df)
    diff = []
    status =[]
    signal = []
    for i in range(len(df['timestamp'])-1):
        diff.append(df.iloc[i]['open'] - df.iloc[i+1]['open'])
        status.append(diff[i]/ df.iloc[i]['open'])
        if status[i] > 0.01:
            signal.append('up')
        elif status[i] < -0.01:
            signal.append('down')
        else:
            signal.append('stay')
    diff.append(1)
    status.append(0)
    signal.append('up')
    df.insert(3,'signal',signal)
    # df.insert(4,'diff',diff)
    print(df)
    df.to_csv('E:\\Queens\\ECE\\2019 WINTER\\ELEC 880\\alphavantage\\venv\\Data'+ '\\' + str(name) +'.csv')
