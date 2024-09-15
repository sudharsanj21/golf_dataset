import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#) title
st.title(':orange[💮classification prediction]🌞')

#) reading the dataset
df1 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\golf_dataset\golf_df.csv")
#) checkbox with df
st.subheader("\n:green[1. dataset🌝]\n")
if (st.checkbox("1.1 original data")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.1 original dataframe]\n")
    data = df1.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))

#) the new data for prediction from user
if (st.checkbox("new data")):
    #)showing original dataframe
    st.markdown("\n#### :blue[1.2 the new data for prediction]\n")
    outlook = st.text_input(":red[**outlook:**]")
    if (outlook):
       temperature  = st.text_input(":red[**temperature:**]")
       if (temperature):
            humidity = st.text_input(":red[**humidity:**]")
            if (humidity):
                windy = st.text_input(":red[**windy:**]")
                if (windy):
                     #) pandas with the new data
                     a_df = pd.DataFrame()
                     a_df['Outlook'] = [outlook]
                     a_df['Temperature'] = [temperature]
                     a_df['Humidity'] = [humidity]
                     a_df['Windy'] = [windy]
                     data = a_df
                     st.dataframe(data.style.applymap(lambda x: 'color:purple'))
                     
                     #)concatenation of dataframes(1-a)
                     df = pd.concat([df1, a_df], axis=0, ignore_index=True)
                     #print(df.tail(4))

                     #) filling target column with 0
                     temp_df = df.fillna(value=0)
                     #temp_df.tail(4)

                     #) dummies
                     cols = ['Outlook','Temperature','Humidity','Windy']
                     temp_df = pd.get_dummies(temp_df,columns=cols,drop_first='True',dtype='int')
                     #print(temp_df)

                     #) extracting the prediction data from dataframe
                     row_list = temp_df.loc[14, :].values.flatten().tolist()
                     #print(row_list)

                     #) removing play data filled by 0 from the list
                     row_list.pop(0)
                     #print(row_list)
                     
                     #) coverting the list to the prescribed format for the prediction
                     new_input = [row_list]
                     #print(new_input)
                     
                     #) dropping again the new input data
                     temp_df.drop([14],axis=0,inplace=True)
                     #print(temp_df)

                     #)logistic regression
                     X = temp_df.drop(['Play'],axis=1)
                     y = temp_df['Play']
                     x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
                     model = LogisticRegression()
                     
                     model.fit(x_train,y_train)
                     train_pred = model.predict(x_train)
                     test_pred = model.predict(x_test)

                     #) the new data for prediction from user
                     if (st.checkbox("ML - classification model")):
                           #)showing original dataframe
                           st.markdown("\n#### :red[2. logistic regression]\n")
                           st.write(f"*******{type(model).__name__}*******")
                           #)training
                           st.code('\n*******Train*******')
                           st.success(f"Accuracy: {accuracy_score(y_train,train_pred)}")
                           
                           #)testing
                           st.code('\n*******Test*******')
                           st.success(f"Accuracy: {accuracy_score(y_test,test_pred)}")

                           #) prediction
                           st.subheader("\n:violet[3. prediction]")
                           if (st.button('prediction')):
                                      #) new input in the format of dummy
                                      st.text("prediction of the given condition")
                                      new_output = model.predict(new_input)
                                      st.info(new_output)
                else:
                        st.error('you have not windy')
            else:
                st.error('you have not entered humidity')
       else:
            st.error('you have not entered temperature')
    else:
         st.error('you have not entered outlook')