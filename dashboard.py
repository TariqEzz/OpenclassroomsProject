import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#chargement données
data_test = open('./X_test.pkl', 'rb') 
X_test= pickle.load(data_test)

#chargement modéle
pickle_in = open('./gr_grid_modele_file.pkl','rb') 
model= pickle.load(pickle_in) 

# traitements des données avec Streamlit :  on teste avec  le id(s) :203725,00 et 187655,00
st.sidebar.number_input("Identifiant__client", key="name")
# trois cas possibles
id_client=st.session_state.name
if  id_client==0:
    st.subheader("Saisir  l'identifiant  à gauche  ")

elif  id_client  not in X_test.SK_ID_CURR.tolist():
    st.subheader("le  client n'existe pas dans notre banque ")
    
else:
    
    for row in X_test.itertuples():
         if row.SK_ID_CURR==id_client:
            dataa={'EXT_SOURCE_3':row.EXT_SOURCE_3,
            'EXT_SOURCE_2':row.EXT_SOURCE_2,
            'FLAG_OWN_CAR':row.FLAG_OWN_CAR,
            'CODE_GENDER':row.CODE_GENDER
             }
            
            prediction=model.predict(X_test[X_test['SK_ID_CURR']==id_client])
            st.write('#  La decision  est:''')
            if prediction == 0 :
                st.success("le  client est elligible pour le crédit")
            else:
                st.warning("le client  est non eligible pour le crédit")
                
            st.write('#  les informations  du client ''')
            st.dataframe(X_test[X_test['SK_ID_CURR']==id_client])  
            st.write('#  les paramètres importants  du client ''')
            st.write(pd.DataFrame(dataa,index=[0]))
            explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values,X_test[X_test['SK_ID_CURR']==id_client],plot_type="bar")
            st.pyplot(fig)
            st.sidebar.write('La classe predicte  est',int(prediction))
            
if st.sidebar.button(" une prediction sur plusieurs clients ", key=None, help=None, on_click=None, args=None, kwargs=None ): 
    
    df=X_test.sample(10)
    st.write('#  Toutes les informations pour chaque client   ''')
    st.write(df)
    st.write('# Paramétres importants et  Resultat des predictions ''')
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values,df,plot_type="bar")
    st.pyplot(fig)
    df['classe prediction']=model.best_estimator_.predict(df)
    st.write(df.loc[:,['SK_ID_CURR','EXT_SOURCE_3','EXT_SOURCE_2','FLAG_OWN_CAR','CODE_GENDER','classe prediction']]) 