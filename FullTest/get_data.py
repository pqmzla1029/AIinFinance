import pandas as pd
import import_functions as ifunc


def main():
    #company='GOOGL'
    #sdate='2005-01-01'
    #edate='2019-03-14'
    company,sdate,edate=ifunc.read_file()
    choice=1
    filename=company+'.csv'
    df=pd.DataFrame()
    df=ifunc.import_data(company,sdate,edate)
    print(df)
    df.to_csv(index=True,index_label='date',path_or_buf=filename)
    """
    if choice==1:
        print()
        df=ifunc.read_full(filename)
    else :
        print()
        df=ifunc.read_spec(filename)
    #print(df)
    """


main()


    
