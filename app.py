import streamlit as st 
import pickle
import numpy as np
import pandas as pd

## Import the model
model=pickle.load(open('model.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
df_x=pickle.load(open('df_x.pkl','rb'))

st.title("Laptop Price Predictor")

### Brand
company=st.selectbox("Brand",df['Company'].unique())
## Model Type               
Type=st.selectbox("Type",df['TypeName'].unique())
## ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
# weight
weight = st.number_input('Weight of the Laptop')  
## Screen Type
Screen=st.selectbox("Touch Screen",['Yes','No'])

IPS_panel=st.selectbox("Panel Type",['Yes','No'])


resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

screen_size = st.number_input('Screen Size')

CPU=st.selectbox("CPU Name",df['Cpu_Name'].unique())

HDD=st.selectbox("Hard Drive(in GB)",[0,128,256,512,1024,2048])

SSD = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

GPU=st.selectbox("GPU Name",df['Gpu brand'].unique())

OS=st.selectbox("Operating System",df['os'].unique())

if st.button('Predict Price'):  

    if Screen=='Yes':
        Screen=1
    else:
        Screen=0

    if IPS_panel=='Yes':
        IPS_panel=1
    else:
        IPS_panel=0
        
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    PPI = ((X_res**2) + (Y_res**2))**0.5/screen_size
        
    data = [[company, Type, ram, weight, Screen, IPS_panel,PPI, CPU, 
                HDD, SSD, GPU, OS]]
        
    new_df = pd.DataFrame(data, columns = ['Company', 'TypeName', 'Ram', 'Weight', 
                                            'Touchscreen', 'IPS_Panel', 'PPI', 'Cpu_Name', 'HDD',
                                            'SSD','Gpu brand','os'])
        
    df_2 = pd.concat([df_x, new_df], ignore_index = True) 
        
    query=pd.get_dummies(columns=['Company','TypeName','Cpu_Name','Gpu brand','os'],drop_first=True,data=df_2)
    st.title("The predicted price of this configuration is " + str(int(np.exp(model.predict(query.tail(1))[0]))))