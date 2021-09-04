
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier


data_test = open('./X_test.pkl', 'rb') 
X_test= pickle.load(data_test)

#st.write(X_test.head())
pickle_in = open('./gr_grid_modele_file.pkl','rb') 
model= pickle.load(pickle_in) 
#st.write('#  les paramètres importants  du client ''')

#st.sidebar.header("Saisir  le numéro  du client ",X_test.head(3).EXT_SOURCE_2)

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
            #st.write(pd.DataFrame(dataa,index=[0]))
            prediction=model.predict(X_test[X_test['SK_ID_CURR']==id_client])
            st.write('#  La decision  est:''')
            if prediction == 0 :
                st.success("le  client est elligible pour le credit")
            else:
                st.warning("le client  est non eligible pour le credit")
                
            st.write('#  les informations  du client ''')
            st.dataframe(X_test[X_test['SK_ID_CURR']==id_client])  
            st.write('#  les paramètres importants  du client ''')
            st.write(pd.DataFrame(dataa,index=[0]))
            
            explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values,X_test[X_test['SK_ID_CURR']==id_client],plot_type="bar")
            st.pyplot(fig)
             # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            #st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))
            #st.sidebar.text("La classe predicte   est:")
            #st.sidebar.int(prediction)
            st.sidebar.write('La classe predicte  est',int(prediction))
            
if st.sidebar.button(" une prediction sur plusieurs client ", key=None, help=None, on_click=None, args=None, kwargs=None ): 
    
    
    df=X_test.sample(10)
    st.write('#  Toutes les informations pour chaque client   ''')
    st.write(df)
    
    st.write('# Parametres importants et  Resultat des predictions ''')
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values,df,plot_type="bar")
    st.pyplot(fig)
    
    df['classe prediction']=model.best_estimator_.predict(df)
    st.write(df.loc[:,['SK_ID_CURR','EXT_SOURCE_3','EXT_SOURCE_2','FLAG_OWN_CAR','CODE_GENDER','classe prediction']]) 
#203725,00
#187655,00