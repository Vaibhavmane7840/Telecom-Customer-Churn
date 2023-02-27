#Import libraries
import pickle
import numpy as np
import pandas as pd
import streamlit as st 
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



#load the model from disk
xgb = pickle.load(open("telecom_leave.pkl", "rb"))

#Import python scripts
st.title('Telecom Customer Churn Prediction')
image = Image.open('pic.jpeg')
st.image(image)
st.subheader('XG Boost Classifier')

st.sidebar.header('User Input Data')

def user_input_features():
    state = st.sidebar.selectbox("State", (('AK','AL','AR','AZ','CA',
                                            'CO','CT','DC','DE','FL',
                                            'GA','HI','IA','ID','IL',
                                            'IN','KS','KY','LA','MA',
                                            'MD','ME','MI','MN','MO',
                                            'MS','MT','NC','ND','NE',
                                            'NH','NJ','NM','NV','NY',
                                            'OH','OK','OR','PA','RI',
                                            'SC','SD','TN','TX','UT',
                                            'VA','VT','WA','WI','WV',
                                            'WY')))
    voice_plan = st.sidebar.radio("Voice plan", ('Yes','No'))
    voice_messages = st.sidebar.number_input ("Insert No. of voice messages (range: 0-52)", 0,52)
    intl_plan = st.sidebar.radio("Intl. plan", ('Yes','No'))
    intl_charge = st.sidebar.number_input ("Insert Intl. charges (range: 0-5.4)", 0.00, 5.40)
    day_charge = st.sidebar.number_input ("Insert day charges (range: 0-59.76)", 0.00, 59.76)
    eve_charge = st.sidebar.number_input ("Insert eve. charges (range: 0-30.91)", 0.00, 30.91)
    night_charge = st.sidebar.number_input ("Insert night charges (range: 0-17.77)", 0.00, 17.77)
    customer_calls = st.sidebar.number_input ("Insert No. of customer calls (range: 0-9)",0, 9)
    data = {'state':state,
            'voice_plan':voice_plan,
            'voice_messages':voice_messages,
            'intl_plan':intl_plan,
            'intl_charge':intl_charge,
            'day_charge':day_charge,
            'eve_charge':eve_charge,
            'night_charge':night_charge,
            'customer_calls':customer_calls}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input Parameters:')
st.write(df.head())

#Preprocess inputs
def preprocess(df):
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})
    # Encode binary categorical features
    binary_list = ['voice_plan','intl_plan']
    df[binary_list] = df[binary_list].apply(binary_map)
    #feature scaling
    LE = LabelEncoder()
    MM = MinMaxScaler()
    df['state'] = LE.fit_transform(df[['state']])
    df['voice_messages'] = MM.fit_transform(df[['voice_messages']])
    df['intl_charge'] = MM.fit_transform(df[['intl_charge']])
    df['day_charge'] = MM.fit_transform(df[['day_charge']])
    df['eve_charge'] = MM.fit_transform(df[['eve_charge']])
    df['night_charge'] = MM.fit_transform(df[['night_charge']])
    return df

df = preprocess(df)

prediction = xgb.predict(df)
proba = xgb.predict_proba(df)
probability = proba*100

if st.button('Predict'):
    if prediction == 1:
        st.warning('Yes, this customer is likely to be churned.')
    else:
        st.success('No, this customer is likely to continue.')
               
if st.button('Prediction Probability'):
    st.write('Probability %:', np.round(probability, 2))  
    













