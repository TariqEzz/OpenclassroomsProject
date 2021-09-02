
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import lightgbm as lgb




data_test = open('./X_test.pkl', 'rb') 
X_test= pickle.load(data_test)

#st.write(X_test.head())
pickle_in = open('./gr_grid_modele_file.pkl','rb') 
model= pickle.load(pickle_in) 
st.write('#  les paramètres importants  du client ''')

#st.sidebar.header("Saisir  le numéro  du client ")

st.sidebar.number_input("Identifiant_client", key="name")


id_client=st.session_state.name

if  id_client==0:
    st.subheader("Saisir  l'identifiant  à gauche  ")

elif  id_client  not in X_test.SK_ID_CURR.tolist():
    st.subheader("le  client n'existe pas dans notre banque ")
else:
    

# You can access the value at any point with:
#st.session_state.name




#prediction=model.predict(df)
#st.subheader("La decision  est:")
#st.write(prediction)

    for row in X_test.itertuples():
         if row.SK_ID_CURR==id_client:
        
        
            dataa={'EXT_SOURCE_3':row.EXT_SOURCE_3,
            'EXT_SOURCE_2':row.EXT_SOURCE_2,
            'FLAG_OWN_CAR':row.FLAG_OWN_CAR,
            'CODE_GENDER':row.CODE_GENDER
             }
            st.write(pd.DataFrame(dataa,index=[0]))
            prediction=model.predict(X_test[X_test['SK_ID_CURR']==id_client])
            st.subheader("La decision  est:")
            if prediction == 0 :
                st.write("le  client est elligible pour le credit")
            else:
                st.write("le client  est non eligible pour le credi")
                
        
#203725,00
#187655,00