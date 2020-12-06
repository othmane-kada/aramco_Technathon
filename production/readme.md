# Aramco Upstream Solutions Technathon 2020 
### task 2: Distributed fiber optic measurement data
# production model for DAS and DTS missed data reconstruction 
 ## hardware requirements : 
 GPU with cuda capable 
 our develepment hardware:
 - ubuntu 20.10
 - ram 32 gb
 - cpu i5-6400 2.7 ghz
 - nvidia geforce GTX 1060 
 ##  software requeriments  
 ### you have to install nvidia rapids from here 
 https://rapids.ai/start.html#get-rapids
 ### the rest of envirements from :
     git clone our project go the requirements dir then :
 ###  with conda : 
     conda create -n dasdts --file req.txt 
 ### with pypi :  
     pip install -r requirements.txt 

 ## how to use
    python main.py -h           -to see help
    python main.py --das_config das.json --dts_config dts.json         -to process DAs and DTS 
    python main.py --das_config das.json                               -to process just DAS
    python main.py --dt_config dts.json                                -to process just DTS
 ## exemple of configs:  
 #### DAS : only input and output are required the params are optional !
   ```json
{
"input":"/aramco/inputs/das",
"output":"aramco/output/das",
"params":{
 "numsimp" : 300,
 "window_size" : 20,
 "test_size" : 0.2,
 "features":3,
 "epochs":50,
 "lstm_size":300,
 "drops":0.2,
 "lr":0.005,
 "batch_size":600
}
}     
```
   #### DTS : only forwrd,backwrd and output are required the params are optional !
   ```json
{
"forwrd":"aramco/inputs/DTSV_data_gapsss.csv",
"backwrd":"aramco/inputs/DTSV_data_gaps22.csv",
"output":"aramco/output/dts",
"params":{
 "n_estimators" : 10,
 "window_size" : 5,
 "test_size" : 0.2,
 "deph_col": "Depth (m)",
 "temp_col": "temperature",
 "time_col": "time"
}
}     
```
#### issues
    not yet))

### contributers:
    Othmane Kada
    Youcef Touahir
    Andrey Mesheryakov
### superviser 
    Timur Zharnikov 
