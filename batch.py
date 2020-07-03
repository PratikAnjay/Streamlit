
import pandas as pd
import pickle
import streamlit as st 
import base64
#import altair as alt

f1 = pd.read_csv('loan_train_data.csv')
x=f1[['Age','Experience','Income','Family','Education','Mortgage','CreditCard']]
y=f1['Personal Loan']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
#y_pred=classifier.predict(X_test)


#pickle_out = open("classifier.pkl","wb")
#pickle.dump(classifier, pickle_out)
#pickle_out.close()

#pickle_in = open("classifier.pkl","rb")
#classifier=pickle.load(pickle_in)


def predict_note_authentication(Age,Experience,Income,Family,Education,Mortgage,CreditCard):
    
    prediction=classifier.predict([[Age,Experience,Income,Family,Education,Mortgage,CreditCard]])
    print(prediction)
    return prediction

def main():
    #st.title("Personal Loan Authenticator")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Personal Loan Eligiblity Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    from PIL import Image
    image_loan=Image.open("LP1.jpg")
    choose_model=st.sidebar.selectbox(label='How would you like to predict?', options=['Online','Batch'])
    st.sidebar.title("Check your Loan Eligiblity")
    st.sidebar.image(image_loan,use_column_width=True)
    if (choose_model=='Online'):
        Age = st.number_input("Age",min_value=1,max_value=100,value=20)
        Experience = st.number_input("Experience",min_value=1,max_value=100,value=3)
        Income = st.text_input("Income","Type Here")
        Family = st.selectbox("Family",[0,1,2,3,4,5,6,7,8,9,10])
        Education = st.selectbox("Education",[1,2,3])
        Mortgage = st.text_input("Mortgage","Type Here")
        CreditCard = st.selectbox("CreditCard",['0','1'])
        result=""
        if st.button("Predict"):
            result=predict_note_authentication(Age,Experience,Income,Family,Education,Mortgage,CreditCard)
            st.success('The output is {}'.format(result))
            st.text("0 : Not Eligible for Personal Loan")
            st.text("1 : Eligible for Personal Loan")
            
    if (choose_model=='Batch'):
        file_upload=st.file_uploader("Upload csv file for Predictions",type=["csv"])
        if file_upload is not None:
            data=pd.read_csv(file_upload)
            predictions=classifier.predict(data)
            data['Prediction'] = predictions
            st.write(data)
            st.text("0 : Not Eligible for Personal Loan")
            st.text("1 : Eligible for Personal Loan")
            
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download Test Set Predictions CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            
                      
if __name__=='__main__':
    main()
    
    
    
