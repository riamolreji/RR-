import streamlit as st
import pickle
import numpy as np

lr = pickle.load(open('lr1_jul21.pkl','rb'))
dt = pickle.load(open('dt1_jul21.pkl','rb'))
rf = pickle.load(open('rf1_jul21.pkl','rb'))


st.title('Insurance Charge Prediction Web App')

st.header('Fill the details to generate Insurnce Charge Prediction')


option = st.sidebar.selectbox('Select the Model',['LineReg','DT_Reg','RF_Reg'])


age = st.slider('Age',18,64)
bmi = st.slider('BMI',15,47)
children = st.selectbox('Children',[0,1,2,3,4,5])
sex = st.selectbox('Gender',['Male','Female'])
smoker = st.selectbox('Smoker',['Yes','No'])
region = st.selectbox('Region',['SE','SW','NE','NW'])


if sex=='Male':
    sex_male = 1
    sex_female = 0
else:
    sex_male = 0
    sex_female = 1

if smoker == "Yes":
    smoker_yes = 1
    smoker_no = 0
else:
    smoker_yes = 0
    smoker_no = 1

if region == "SE":
    region_northwest =0
    region_northeast =0
    region_southwest =0
    region_southeast =1
elif region == "SW":
    region_northwest =0
    region_northeast =0
    region_southwest =1
    region_southeast =0
elif region == "NW":
    region_northwest =1
    region_northeast =0
    region_southwest =0
    region_southeast =0
else:
    region_northwest =0
    region_northeast =1
    region_southwest =0
    region_southeast =0

test = [age,bmi,children,sex_male,smoker_yes,region_northwest,
        region_southeast,region_southwest]
test = np.array(test).reshape(1,8)


if st.button('Predict'):
    if option=="LineReg":
        st.success(lr.predict(test)[0])
    elif option=="DT_Reg":
        st.success(dt.predict(test)[0])
    else:
        st.success(rf.predict(test)[0])








