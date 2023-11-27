import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
st.title('Traffic Volume Prediction: A Machine Learning App')
st.image('traffic_pic.jpeg')

#read in pickled dt file
dt_pickle = open('deicision_tree_reg_traffic_hour_cat_fixed.pickle', 'rb') 
bestRegTree_traffic = pickle.load(dt_pickle) 
dt_pickle.close() 

#read in pickled rf file
regRF_pickle = open('reg_RF_traffic_fixed.pickle', 'rb')
bestRegRF_traffic = pickle.load(regRF_pickle)
regRF_pickle.close()

#read in pickled adaBoost file
adaBoost_pickle = open('reg_AdaBoost_fixed.pickle', 'rb')
bestAdaBoost_traffic = pickle.load(adaBoost_pickle)
adaBoost_pickle.close()

#read in pickled xgBoost
xgboost_pickle = open('midtermXgBoost_hour_cat_fixed.pickle', 'rb')
bestXgBoost_traffic = pickle.load(xgboost_pickle)
xgboost_pickle.close()

#read in dataset
df_traffic_volume = pd.read_csv('Traffic_Volume.csv')
df_traffic_volume.drop('weather_description',axis=1,inplace=True) #drop unuse column
df_traffic_volume['holiday'] = df_traffic_volume['holiday'].fillna('None') #replace NaN w/ "None" string
df_traffic_volume['date_time'] = pd.to_datetime(df_traffic_volume['date_time']) #convert to datatime object
df_traffic_volume['month'] = df_traffic_volume['date_time'].dt.month_name() #extract month name
df_traffic_volume['day'] = df_traffic_volume['date_time'].dt.day_name() #extract days of the week
df_traffic_volume['time'] = df_traffic_volume['date_time'].dt.hour #extract hour

df_traffic_volume.drop('date_time',axis=1,inplace=True) #drop unuse column

#create input options
with st.form('user_inputs'): 
  holiday = st.selectbox('Choose whether today is a designated holiday or not', options=df_traffic_volume['holiday'].unique()) 
  tempK = st.number_input('Average temperature in Kelvin')
  rain = st.number_input('Amount in mm of rain that occurred in the hour')
  snow = st.number_input('Amount in mm of snow that occurred in the hour')
  cloud = st.number_input('Percentage of cloud cover')
  weather = st.selectbox('Choose the current weahter', options =df_traffic_volume['weather_main'].unique())
  month = st.selectbox('Choose month', options=df_traffic_volume['month'].unique())
  day = st.selectbox('Choose day of the week', options=df_traffic_volume['day'].unique())
  hour = st.selectbox('Choose hour', options=df_traffic_volume['time'].unique())
  model = st.selectbox('Select Machine Learning Model For Prediction',options=['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost'])
  st.form_submit_button()

#concate use input
row_user_input = [holiday, tempK, rain, snow, cloud, weather, month, day, hour]

#data preprocessing
df_traffic_volume = df_traffic_volume.drop(columns=['traffic_volume'])
df_traffic_volume.loc[len(df_traffic_volume)] = row_user_input

#dropna
df_traffic_volume.dropna()
 
#convert categorical variables to dummies
X_traffic_cate_var = ['holiday','weather_main','month','day','time']
X_traffic_encoded = pd.get_dummies(df_traffic_volume, columns = X_traffic_cate_var)
user_row = X_traffic_encoded.tail(1)


# Creating a 4x2 DataFrame

pd.set_option('display.precision', None)


data = {
    'ML Model': ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost'],
    'R-squared': [0.928404,0.940014, 0.338440, 0.952784],  # R-squared values
    'RMSE': [530.091387, 485.212079, 1611.354330, 430.476091]  # RMSE values
}
df_models = pd.DataFrame(data)


def highlight_row(s):
    if s['ML Model'] == 'Random Forest':
        return ['background-color: lime'] * len(s)
    elif s['ML Model'] == 'AdaBoost':
        return ['background-color: orange'] * len(s)
    else:
        return [''] * len(s)


highlighted_df = df_models.style.apply(highlight_row, axis=1)


st.write('These ML models exhibited the following predictive performance on the test dataset')
st.dataframe(highlighted_df)



if model == 'Decision Tree':
   
   st.write('Decision Tree Traffic Prediction:')
   pred_user = bestRegTree_traffic.predict(user_row)
   st.write(int(pred_user))
   st.image('midterm2_dt_hour_cat_fixed.png')

elif model == 'Random Forest':
   
   st.write('Random Forest Traffic Prediction:')
   pred_user_rf = bestRegRF_traffic.predict(user_row)
   st.write(int(pred_user_rf))
   st.image('feature_importance_rf_fixed_hour_cat.svg')

elif model == 'AdaBoost':
   
   st.write('adaBoost Traffic Prediction:')
   pred_user_adaBoost = bestAdaBoost_traffic.predict(user_row)
   st.write(int(pred_user_adaBoost))
   st.image('adaBoost_feature_imp_hour_cat_fixed.svg')
   
elif model == 'XGBoost':
   
   st.write('xgBoost Traffic Prediction:')
   pred_user_xgBoost = bestXgBoost_traffic.predict(user_row)
   st.write(int(pred_user_xgBoost))
   st.image('midterm2_xgb_hour_cat_fixed.svg')
  

