import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from jobs.das import das
from jobs.dts import dts
import os
import time
import argparse
import json
import os.path
class dasdts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='das and dts data recover')
        self.parser.add_argument('--das_config', type=str,
                        help='json file containing das configuration , inputs outputs and training parametrs')
        self.parser.add_argument('--dts_config', type=str,
                        help='json file containing dts configuration , inputs outputs and training parametrs')
    def run(self):
        print('starting initialisation')
        start_time = time.time()
        self.__init__()
        print("initialisation time is is {} seconds".format(time.time()-start_time))
        print('starting configs verification')
        self.verify_config_files(self.parser)

    def read_file_content(self,path):
        if os.path.isfile(path):
            with open(path) as f:
                file_content = f.read()
            return file_content
        else :
            return []
    def read_config(self,arg):
        if len(self.read_file_content(arg)) > 0:
            file =self.read_file_content(arg)
            config = json.loads(file, strict=False)
            return config
        else:
            return []
    def verify_config_files(self,parser):
            results={}
            args  = parser.parse_args()
            if args.dts_config!=None:
                results['dts_config']=self.read_config(args.dts_config) if len(self.read_config(args.dts_config))>0 else {}
                results['dts_config']=self.dts_configs(results['dts_config'])
            if args.das_config != None:
                results['das_config']=self.read_config(args.das_config) if len(self.read_config(args.das_config))>0 else {}
                results['das_config']=self.das_configs(results['das_config'])
            if  len(results)==0:
                print('none of the configs was sets please add config files')
                parser.print_help()
            else:
                if len(results)==1:
                    if 'dts_config' in results:
                        print('DAs config was not set only DTS will be processed')
                        print('starting DTS recovering job')
                        start_time = time.time()
                        dts(results['dts_config']).run()
                        print(" DTS recovering job time is is {} seconds".format(time.time() - start_time))
                    else:
                        print('DTS config was not set only DAS will be processed')
                        print('starting DAS recovering job')
                        start_time = time.time()
                        das(results['das_config']).run()
                        print(" DAS recovering job time is is {} seconds".format(time.time() - start_time))
                else :
                    print('DTS and DAS will be processed')
                    print('starting DTS recovering job')
                    start_time = time.time()
                    dts(results['dts_config']).run()
                    print(" DTS recovering job time is is {} seconds".format(time.time() - start_time))
                    print('starting DAS recovering job')
                    start_time = time.time()
                    das(results['das_config']).run()
                    print(" DAS recovering job time is is {} seconds".format(time.time() - start_time))

            return results

    def dts_configs(self,config):
        results = {}
        if 'forwrd' in config and 'backwrd' in config and 'output' in config:
            if os.path.exists(config['forwrd']) and os.path.exists(config['backwrd']) :
                if not os.path.exists(config['output']):
                    os.makedirs(config['output'])
                params = {'n_estimators': 10, 'window_size': 5, 'test_size': 0.2, 'deph_col': 'Depth (m)', 'temp_col': 'temperature',
                          'time_col': 'time'}
                for i in params:
                    if i in config['params']:
                        params[i] = config['params'][i]
                results['forwrd'] = config['forwrd']
                results['backwrd'] = config['backwrd']
                results['output'] = config['output']
                results['params'] = params
            else:
                print('please verify that forwrd and backwrd files exists!!!!')
        else:
            print('please verify that forwrd and backwrd and output are in config!!!!')
        return results

    def das_configs(self, config):
        results={}
        if 'input'  in config and 'output' in config :
            if os.path.exists(config['input']):
                if not os.path.exists(config['output']):
                    os.makedirs(config['output'])
                params = {'numsimp':300, 'window_size':20, 'test_size':0.2, 'features':3, 'epochs':50,
                                'lstm_size':300, 'drops':0.2, 'lr':0.005, 'batch_size':600}
                for i in params:
                    if i in config['params']:
                        params[i] = config['params'][i]
                results['input']=config['input']
                results['output']=config['output']
                results['params']=params
            else:
                print('please verify that input files exists!!!!')
        else:
            print('please verify that input and output are in config!!!!')
        return results

dasdts().run()
