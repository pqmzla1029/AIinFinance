import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
import pandas as pd

def import_data(company,sdate,edate):
    '''
    company='GOOGL'
    sdate='2016-01-01'
    edate='2019-03-14'
    '''
    data = yf.download(company,sdate,edate)
    path=company+'.csv'
    data.to_csv(index=True,index_label='date',path_or_buf=path)
    data.Close.plot()
    #plt.show()

def read_full(filename):
    df=pd.read_csv(filename)
    return df
        
def read_spec(filename):
    df=pd.read_csv(filename)
    df = df[['Close']]
    return df

def main():
    company='AAPL'
    sdate='2016-01-01'
    edate='2019-03-14'
    choice=1
    filename=company+'.csv'
    df=pd.DataFrame()
    import_data(company,sdate,edate)
    if choice==1:
        print()
        df=read_full(filename)
    else :
        print()
        df=read_spec(filename)
    print(df)
    

main()
    
