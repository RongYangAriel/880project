import pandas as pd

df1 = pd.read_csv('AAPL.csv')
df2 = pd.read_csv('AMAZ.csv')
df3 = pd.read_csv('FB.csv')
df4 = pd.read_csv('GOOGL.csv')
df5 = pd.read_csv('MSFT.csv')

df = pd.DataFrame()

df = df.append(df1).append(df2).append(df3).append(df4).append(df5).drop(['Unnamed: 0'], axis=1)
df.to_csv('pass_to_bert.txt')
# print(df.head())

down_df = df.signal.str.count('down').sum()
print('down', down_df)
up_df = df.signal.str.count('up').sum()
print('up', up_df)
stay_df = df.signal.str.count('stay').sum()
print('stay', stay_df)
