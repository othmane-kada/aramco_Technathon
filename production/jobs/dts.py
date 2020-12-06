import pandas as pd
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from functools import partial
import time
class dts :
    def __init__(self,config):
        self.deph_col = config['params']['deph_col']
        self.temp_col = config['params']['temp_col']
        self.time_col = config['params']['time_col']
        self.window_size = config['params']['window_size']
        self.test_size = config['params']['test_size']
        self.n_estimators = config['params']['n_estimators']
        self.output=config['output']
        self.forwrd = pd.read_csv(config['forwrd'])
        self.forwrd = self.forwrd.melt(id_vars=[self.deph_col], var_name=self.time_col, value_name=self.temp_col)
        self.backwrd = pd.read_csv(config['backwrd'])
        self.backwrd = self.backwrd.melt(id_vars=[self.deph_col], var_name=self.time_col, value_name=self.temp_col)

    def run(self):
        print('starting initialisation')
        start_time = time.time()
        print("initialisation time is is {} seconds".format(time.time()-start_time))
        print('starting preprocessing')
        start_time = time.time()
        self.data=self.preprocess(self.forwrd)
        print("preprocessing time is is {} seconds".format(time.time()-start_time))
        print('starting training')
        start_time = time.time()
        self.model=self.train(self.data)
        print("training time is is {} seconds".format(time.time()-start_time))
        print('starting prediction and saving results')
        start_time = time.time()
        resultforwrd=self.results(self.forwrd)
        resultbackwrd=self.results(self.backwrd)
        resultbackwrd.pivot_table(index='Depth (m)',
                           columns='time',
                           values='temperature').to_csv(self.output+'/resultbackwrd.csv')
        resultforwrd.pivot_table(index='Depth (m)',
                           columns='time',
                           values='temperature').to_csv(self.output+'/resultforwrd.csv')
        print("prediction time  is is {} seconds".format(time.time()-start_time))
        return resultforwrd,resultbackwrd

    def rollback(self,data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def processInput(self,i):
        forwrd=self.forwrd
        deph_col=self.deph_col
        temp_col=self.temp_col
        window_size=self.window_size
        return self.rollback(forwrd[forwrd[deph_col]==i][[temp_col]], n_in=window_size, n_out=1, dropnan=True)

    def preprocess(self,df):
        forwrd=df
        deph_col=self.deph_col
        num_cores = multiprocessing.cpu_count()
        data=pd.DataFrame()
        df_chunks=forwrd[deph_col].drop_duplicates()
        with multiprocessing.Pool(num_cores) as pool:
            data = pd.concat(pool.map(self.processInput, df_chunks), ignore_index=True)
        return data

    def train(self,data) :
        data=data.astype('float32')
        n_estimators=self.n_estimators
        test_size=self.test_size
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['var1(t)'],axis=1),data[['var1(t)']], test_size=test_size, random_state=42)
        interpolation = RandomForestRegressor(n_estimators=n_estimators,n_jobs=-1)
        interpolation.fit(X_train,y_train)
        print('score for testing',interpolation.score(X_test,y_test))
        return interpolation

    def predicter(self,input_df,depth,model):
        time_col=self.time_col
        deph_col=self.deph_col
        temp_col=self.temp_col
        window_size=self.window_size
        tmp=input_df[input_df[deph_col]==depth][temp_col].reset_index(drop=True).astype('float32').values.tolist().copy()
        for i in sum(np.argwhere(np.isnan(tmp)).tolist(),[]):
            tmp[i]=model.predict(np.array(tmp[i-window_size:i]).reshape(1, -1))[0]
        result=pd.DataFrame(tmp,columns=[temp_col])
        result[deph_col]=depth
        result[time_col]=input_df[input_df[deph_col]==depth][time_col].reset_index(drop=True).values
        return result

    def predicter_paralel(self,i,df_input):
            return self.predicter(df_input, i, model=self.model)

    def results(self,df_input):
        deph_col=self.deph_col
        num_cores = multiprocessing.cpu_count()
        df_input_results = pd.DataFrame()
        df_chunks = df_input[deph_col].drop_duplicates()
        with multiprocessing.Pool(num_cores) as pool:
            df_input_results=pd.concat(pool.map(partial(self.predicter_paralel, df_input=df_input), df_chunks), ignore_index=True)
        return df_input_results