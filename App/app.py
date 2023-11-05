#importing standard libraries
import streamlit as st
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_path=os.path.dirname(os.getcwd())+'\App_Subscription_Analyser\Models\svc_model.pkl'
model=joblib.load(model_path)
scale_path=os.path.dirname(os.getcwd())+'\App_Subscription_Analyser\Models\scale.pkl'
scale=joblib.load(scale_path)
df=pd.read_csv(os.path.dirname(os.getcwd())+'\App_Subscription_Analyser\Dataset\predicted_result.csv')
@st.cache_data
def view_analytics():
    st.dataframe(df.style.applymap(color_val),width=1500)

def color_val(val):
    if(val==0):
        return 'background-color:red'
    elif(val==1):
        return 'background-color:green'
    
comment="""
def predict():
    with st.form('query'):
            userid=st.text_input('Please enter userid:')
            phnnum=st.radio('Verified PhoneNumber?',['Yes','No'])
            dob=st.radio('Verified DOB',['Yes','No'])
            loc=st.radio('Location verified?',['Yes','No'])
            weekend=st.radio('Is today a weekend?',['Yes','No'])
            country=st.radio('Country verified?',['Yes','No'])
            credit=len(st.multiselect('Credit Screens visited:',['Credit3Container', 'Credit3Dashboard', 'Credit3', 'Credit1', 'Credit2']))
            numscreen=st.number_input('Number of screens visited:')
            bank=st.radio('Bank verification done?',['Yes','No'])
            idscreen=st.radio('ID Screen ready:',['Yes','No'])
            mobnum=st.radio('Mobile Number Verified?',['Yes','No'])
            loan=len(st.multiselect('Loan screen visited:',['Loan2', 'Loan3', 'Loan', 'Loan4']))
            alert=st.radio('Alert generation allowed?',['Yes','No'])
            age=st.slider('Age of customer')
            other=st.number_input('Number of other screens visited')
            query=[phnnum,other,dob,loc,weekend,country,credit,numscreen,bank,idscreen,mobnum,loan,alert,age]
            submit=st.form_submit_button("Submit")
            if(submit):
                return query
"""





st.title("App Subscription Analyser")
###['enrolled', 'VerifyPhone', 'Other', 'VerifyDateOfBirth', 'location',
###      'is_weekend_enrolleddate', 'VerifyCountry', 'Credit', 'numscreens',
 ###      'BankVerification', 'idscreen', 'VerifyMobile', 'Loan_all', 'Alerts',
 ###      'age', 'user']


st.write("""
In this project, we use advanced data analytics and predictive modeling to boost customer engagement and increase subscription rates for a mobile app. We analyze user behavior to identify patterns and triggers that encourage premium subscriptions and in-app purchases.""")
color_mapping={0:'red',1:'green'}
page=st.sidebar.selectbox("""
Activity:""",("View Analytics","New Prediction"))
if(page=="View Analytics"):
    count_df=df['Enrolled?'].value_counts().reset_index()
    st.write(count_df)
    st.bar_chart(count_df.set_index('Enrolled?')['count'])
    view_analytics()
elif(page=="New Prediction"):
        userid = st.text_input('Please enter userid:')
        phnnum = 1 if(st.radio('Verified PhoneNumber?', ['Yes', 'No']) == 'Yes') else 0
        dob = 1 if(st.radio('Verified DOB', ['Yes', 'No']) == 'Yes') else 0
        loc = 1 if(st.radio('Location verified?', ['Yes', 'No']) == 'Yes') else 0
        weekend = 1 if(st.radio('Is today a weekend?', ['Yes', 'No']) == 'Yes') else 0
        country = 1 if(st.radio('Country verified?', ['Yes', 'No']) == 'Yes') else 0
        credit = len(st.multiselect('Credit Screens visited:', ['Credit3Container', 'Credit3Dashboard', 'Credit3', 'Credit1', 'Credit2']))
        numscreen = st.number_input('Number of screens visited:',step=1)
        bank = 1 if(st.radio('Bank verification done?', ['Yes', 'No']) == 'Yes') else 0
        idscreen = 1 if(st.radio('ID Screen ready?', ['Yes', 'No']) == 'Yes') else 0
        mobnum = 1 if(st.radio('Mobile Number Verified?', ['Yes', 'No']) == 'Yes') else 0
        loan = len(st.multiselect('Loan screen visited:', ['Loan2', 'Loan3', 'Loan', 'Loan4']))
        alert = 1 if(st.radio('Alert generation allowed?', ['Yes', 'No']) == 'Yes') else 0
        age = st.slider('Age of customer')
        other = st.number_input('Number of other screens visited',step=1)
        query = [phnnum, other, dob, loc, weekend, country, credit, numscreen, bank, idscreen, mobnum, loan, alert, age]
        scaled_query=scale.transform([query])


        if st.button("Predict"):
            #st.write(scaled_query)
            result = model.predict(scaled_query)
            prob=model.predict_proba(scaled_query)
            if(result==1):
                st.success(f'The customer is a potential buyer of subscription. The subscription score is {prob[0][1]*100}')
            elif(result==0):
                st.error(f'The customer is not a potential buyer of subscription. The subscription score is {prob[0][1]*100}')

    

        

            
        
        





