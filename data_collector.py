import requests
import pandas as pd
import yfinance as yf


def yahoo_retriever():
    #download fear and greed index
    r = requests.get('http://api.alternative.me/fng/?limit=0')
    df_fng = pd.DataFrame(r.json()['data'])
    df_fng.value = df_fng.value.astype(int)
    df_fng.timestamp = pd.to_datetime(df_fng.timestamp,unit = 's')
    df_fng.set_index('timestamp',inplace = True)
    df_fng = df_fng[::-1]

    #download bitcoin
    df_btc = yf.download('BTC-USD')
    df_btc = df_btc.drop(['Open', 'High','Low','Adj Close','Volume'], axis = 1)
    df_btc.index.name = 'timestamp'
    df_btc = df_btc.rename(columns = {'Close': 'BTC'}, inplace = False)

    #download Etherium
    df_eth = yf.download('ETH-USD')
    df_eth.index.name = 'timestamp'
    merged = df_eth.merge(df_btc,on = 'timestamp')
    merged = merged.merge(df_fng,on = 'timestamp')
    merged = merged.drop(['value'], axis=1)
    merged = merged.drop(['value_classification', 'time_until_update'], axis = 1)
    #merged = merged.rename(columns={'value': 'FNG'}, inplace=False)
    merged = merged.drop(['Adj Close'], axis = 1)
    return merged






