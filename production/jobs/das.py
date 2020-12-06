import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from cuml.decomposition import PCA as cuPCA
from cuml.neighbors import NearestNeighbors
from functools import partial
import time
import glob
import h5py
import swifter
import sys
from scipy import signal
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.optimizers import  Adam,RMSprop
import dask.array as da

class das :
    def __init__(self,config):
        self.numsimp = config['params']['numsimp']
        self.window_size = config['params']['window_size']
        self.test_size = config['params']['test_size']
        self.features=config['params']['features']
        self.epochs=config['params']['epochs']
        self.lstm_size=config['params']['lstm_size']
        self.drops=config['params']['drops']
        self.lr=config['params']['lr']
        self.batch_size=config['params']['batch_size']
        self.das_directory =config['input']
        self.rep =config['output']
        self.files = glob.glob(self.das_directory + "/*.h5")
        if self.files ==0 :
            print('no h5 das files in the input folder , please rewrite DAS configs')
        self.filesdf = pd.DataFrame(self.files, columns=['file'])
        self.chanels = list(self.h5_reader(self.filesdf['file'][0])['channel'])
        self.filesdf['name'] = self.filesdf['file'].swifter.apply(lambda x: x.split('/')[-1].replace('.h5', ''))
        self.chanels = list(self.h5_reader(self.filesdf['file'][0])['channel'])
        self.all_files=self.filenames_finder(self.filesdf)

    def run(self):
        print('starting initialisation')
        start_time = time.time()
        print("initialisation time is is {} seconds".format(time.time()-start_time))
        print('starting preprocessing')
        start_time = time.time()
        self.data_by_chanel=self.data_bychanel_creater()
        self.dataset = np.concatenate([self.signal_transform(self.data_by_chanel, i) for i in range(len(self.chanels))], axis=0)
        self.all_files = self.all_files.merge(self.data_by_chanel, how='left', on='file')
        print("preprocessing time is is {} seconds".format(time.time()-start_time))
        print('starting training')
        start_time = time.time()
        self.model=self.preprocessing_training()
        print("training time is is {} seconds".format(time.time()-start_time))
        print('starting prediction and saving results')
        start_time = time.time()

        for i in self.all_files[self.all_files['output'] == 1].index:
            self.predicter(i)
        self.output = self.all_files[self.all_files['output'] == 1].copy().reset_index(drop=True)
        self.extract = self.all_files[self.all_files['output'] == 0].copy().reset_index()
        for ch in range(len(self.chanels)):
            self.get_nearest(ch)
        print("prediction time  is is {} seconds".format(time.time() - start_time))
        start_time = time.time()
        print('starting writing results')
        self.filesdf['xarray'] = self.filesdf['file'].swifter.apply(lambda x: da.from_array(self.das_array_extracter(x)))
        print('preparation finished !!')
        self.output.swifter.apply(lambda x: self.generating_missed_files(x), axis=1)
        print("writing results time  is is {} seconds".format(time.time()-start_time))
        return self.model

    def h5_reader(self,f):
        return h5py.File(f, 'r')

    def time_extracter(self,h5file):
        return int(h5file['t'][0])

    def das_extracter(self,f):
        return pd.DataFrame(np.array(self.h5_reader(f).get('das')))

    def das_array_extracter(self,f):
        return self.h5_reader(f).get('das')

    def filename_extracter(self,last_file_name):
        if last_file_name[-2:] == '17':
            return str(int(last_file_name) + 30)
        else:
            return str(int(last_file_name) + 70)

    def filenames_finder(self,filesdf):
        filesdf['start'] = filesdf['file'].swifter.apply(lambda x: self.time_extracter(self.h5_reader(x)))
        filesdf = filesdf.sort_values('start').reset_index(drop=True)
        all_files = pd.DataFrame(list(range(filesdf.start.min(), filesdf.start.max(), 30)), columns=['start']).copy()
        all_files = all_files.merge(filesdf, how='left', on='start')
        all_files['last_file_name'] = all_files['name'].shift(periods=1)
        all_files = all_files.fillna('-1')
        all_files['output'] = 0
        for i in all_files[all_files.name == '-1'].index:
            all_files.loc[i, 'name'] = self.filename_extracter(all_files.loc[i, 'last_file_name'])
            all_files.loc[i + 1, 'last_file_name'] = all_files.loc[i, 'name']
            all_files.loc[i, 'output'] = 1
        all_files[all_files.output == 1][['start', 'name']]
        return all_files

    def signal_preprocessing(self,sig):
        lowpass_filter = signal.butter(1, Wn=0.001, btype='low', output='sos')
        a, b = signal.welch(signal.resample((signal.sosfilt(lowpass_filter, sig)), int(self.numsimp)), fs=0.1,
                            scaling='spectrum')
        return b

    def preprocessing_generale(self,file):
        data = []
        tmp = self.das_extracter(file)
        for i in tmp.columns:
            data.append(self.signal_preprocessing(tmp[i]))
        result = pd.DataFrame([data])
        result['file'] = file
        return result

    def data_bychanel_creater(self):
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cores) as pool:
            data_by_chanel = pd.concat(pool.map(self.preprocessing_generale, self.filesdf['file']), ignore_index=True)
        return data_by_chanel

    def signal_transform(self,filesdf, chanel):
        scaler = MinMaxScaler()
        pca_cuml = cuPCA(n_components=self.features, random_state=42)
        result_cuml = pca_cuml.fit_transform(scaler.fit_transform(pd.DataFrame(filesdf[chanel].values.tolist())))
        self.data_by_chanel[chanel] = pd.DataFrame(
            pd.DataFrame(result_cuml).apply(lambda r: tuple(r), axis=1).apply(np.array))
        return result_cuml

    def shifter(self,dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            dataX.append(a)
            dataY.append(dataset[i + look_back, :])
        x = np.asarray(dataX).reshape(len(dataX), look_back, len(dataY[1]))
        y = np.asarray(dataY).reshape(len(dataY), len(dataY[1]))
        return train_test_split(x, y, test_size=self.test_size, random_state=42)

    def preprocessing_training(self):
        window =self.window_size
        x_train, x_test, y_train, y_test = self.shifter(self.dataset, window)
        model = self.trainer(window, x_train, x_test, y_train, y_test)
        return model

    def trainer(self,window, x_train, x_test, y_train, y_test):
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        rm = RMSprop(
            learning_rate=self.lr,
        )
        shape = self.features
        size = self.lstm_size
        drops = self.drops
        model = Sequential()
        model.add(LSTM(size, input_shape=(window, shape), return_sequences=True))
        model.add(Dropout(drops))
        model.add(LSTM(size, return_sequences=False))
        model.add(Dropout(drops))
        model.add(Dense(shape))
        model.compile(loss='mae', optimizer=rm, metrics=['cosine_similarity'])
        model.fit(x_train, y_train, epochs=self.epochs, verbose=2, validation_split=self.test_size, batch_size=self.batch_size)
        print('model score : ', model.evaluate(x_test, y_test, verbose=2)[1])
        return model

    def extract_rows(self,end):
        window=self.window_size
        features=self.features
        all_results = []
        segment = self.all_files[end - window:end][range(len(self.chanels))].copy().reset_index(drop=True)
        for ch in range(len(self.chanels)):
            try:
                all_results.append(
                    np.concatenate([segment[ch][i] for i in range(len(segment[ch]))], axis=0).reshape(1, window,
                                                                                                      features))
            except:
                print(segment)

        return np.concatenate([i for i in all_results], axis=0)

    def restorer(self,chanel, i, resutls):
        self.all_files[chanel].loc[i] = resutls[chanel].copy()

    def predicter(self,end):
        resutls = self.model.predict(self.extract_rows(end))
        for ch in range(len(self.chanels)):
            self.restorer(ch, end, resutls)

    def get_nearest(self,chanel):
        NN = NearestNeighbors(n_neighbors=1)
        NN.fit(pd.DataFrame(self.extract[chanel].values.tolist()))
        self.output[chanel] = NN.kneighbors(pd.DataFrame(self.output[chanel].values.tolist()), return_distance=False)

    def generating_missed_files(self,exemple, dest, t):
        rep = self.rep+'/'
        dest = rep + dest + '.h'
        with h5py.File(exemple, 'r') as f1:
            with h5py.File(dest, 'w') as f2:
                for ds in f1.keys():
                    f1.copy(ds, f2)
                f2['t'][:] = np.array(range(t, t + 30000))
        f2.close()
        f1.close()
        return dest

    def das_collecter(self,x):
        return da.stack([self.filesdf['xarray'][x[i]][:,i] for i in range(384) ], axis=1).compute()

    def generating_missed_files(self,x):
        rep = self.rep + '/'
        dest = rep +x['name'] + '.h'
        with h5py.File(self.filesdf['file'][0], 'r') as f1:
            with h5py.File(dest, 'w') as f2:
                for ds in f1.keys():
                    f1.copy(ds, f2)
                f2['t'][:] = np.array(range(x['start'], x['start'] + 30000))
                f2['das'][:] = self.das_collecter(x)
        f2.close()
        f1.close()
