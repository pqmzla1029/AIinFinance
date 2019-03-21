import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
import pandas as pd

def import_data(company,sdate,edate):
    '''
    company='GOOGL'
    sdate='2000-01-01'
    edate='2019-03-14'
    '''
    data = yf.download(company,sdate,edate)
    path=company+".csv"
    print(path)
    #data.to_csv(index=True,index_label='date',path_or_buf=path)
    data.Close.plot()
    return data
    #plt.show()

def read_full(filename):
    df=pd.read_csv(filename)
    return df
        
def read_spec(filename):
    df=pd.read_csv(filename)
    df = df[['Close']]
    return df


def read_file():
    text_file=open("input.txt","r+")
    lines = text_file.read().split(' ')
    #print(lines[0]+" "+lines[1]+" "+lines[3])
    return lines[0],lines[1],lines[3]
    text_file.close()




    
