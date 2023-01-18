
#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Based on multiple machine learning to predict the lymph node metastasis and prognosis of conjunctival melanoma')
st.title('Based on multiple machine learning to predict the lymph node metastasis and prognosis of conjunctival melanoma')

#%%set variables selection
st.sidebar.markdown('## Variables')
Primary_Site = st.sidebar.selectbox('Primary_Site',('Conjunctiva','Choroid','Ciliary body'),index=1)
Marital_status = st.sidebar.selectbox("Marital_status",('Married','Unmarried','Unkonwn'),index=1)
Laterality = st.sidebar.selectbox("Laterality",('Left','Right','Other'),index=1)
Tumor_size = st.sidebar.selectbox("Tumor_size (mm)",('<11mm','≥11mm'),index=1)
Radiation = st.sidebar.selectbox("Radiation", ("No","Yes"),index = 1)
Chemotherapy = st.sidebar.selectbox("Chemotherapy", ("No","Yes"),index = 1)
T = st.sidebar.selectbox("T stage", ("T1","T2","T3","T4","TX"), index = 1)


#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Conjunctiva':1,'Choroid':2,'Ciliary body':3,
       'Married':1, 'Unmarried':2,'Unkonwn':3,
       'Left':1,'Right':2,'Other':3,
       '<11mm':0,'≥11mm':1,
       "No":1,"Yes":2,
       "T1":1,"T2":2,"T3":3,"T4":4,"TX":5}
Primary_Site =map[Primary_Site]
Marital_status = map[Marital_status]
Laterality = map[Laterality]
Tumor_size = map[Tumor_size]
Radiation = map[Radiation]
Chemotherapy = map[Chemotherapy]
T = map[T]
# 数据读取，特征标注
#%%load model
mlp_model = joblib.load(r'E:\大五寒假相关文件\结膜黑色素瘤\mlP_tumor_eye_model.pkl')

#%%load data
hp_train = pd.read_csv(r'E:\大五寒假相关文件\结膜黑色素瘤\tuomor_data.csv')
features =["Primary_Site","Marital_status","Laterality","Tumor_size","Radiation",'Chemotherapy',"T"]
target = 'N'
y = np.array(hp_train[target])
sp = 0.5

is_t = (mlp_model.predict_proba(np.array([[Primary_Site,Marital_status,Laterality,Tumor_size,Radiation,Chemotherapy,T]]))[0][1])> sp
prob = (mlp_model.predict_proba(np.array([[Primary_Site,Marital_status,Laterality,Tumor_size,Radiation,Chemotherapy,T]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Lymph Node Metastasis'
else:
    result = 'Low Risk Lymph Node Metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Lymph Node Metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Lymph Node Metastasis group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[Primary_Site,Marital_status,Laterality,Tumor_size,Radiation,Chemotherapy,T]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0
    
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = mlp_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of MLP model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of MLP model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of MLP model')
    mlp_prob = mlp_model.predict(X)
    cm = confusion_matrix(y, mlp_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NLNM', 'LNM'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of MLP model")
    disp1 = plt.show()
    st.pyplot(disp1)