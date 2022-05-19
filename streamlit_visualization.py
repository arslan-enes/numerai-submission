import pandas as pd
import streamlit as st
import numpy as np


def stream():
    header = st.container()
    dataset = st.container()
    features = st.container()
    model_training = st.container()

    with header:
        st.title("Numerai Veri Seti ile Makine Öğrenmesi Modeli Oluşturma")
        st.write("""Numerai, founded in 2015, runs a weekly data science tournament at https://numer.ai. 
    Data scientists fromaround the world can download clean, obfuscated data for free, build a model and submit to predict the stock 
    market.Because it’s obfuscated data, you don’t have to know anything about finance to participate. Participants have skin in 
    the game — a stake applied to their predictions. Good predictions are rewarded with more crypto (Numeraire) and bad predictions 
    are burned (the crypto stake is destroyed forever).""")
        st.header("Numeraire (NMR) token nedir ?")
        st.write("""Numeraire (NMR) Token: An ERC-20-based token which is used to stake on Numerai. NMR can be earned by competing 
    in the Numerai tournament and by making technical contributions to Numerai.""")
    
    with dataset:
        sel_col , disp_col = st.columns(2)
        target = train['target']
        st.header("Veri Seti Hakkında")
        rows = sel_col.slider('Gösterilmesini istediğiniz satır sayısını seçiniz:',min_value = 10, max_value = 100, value=20, step = 10)
        st.write(train.head(rows))
        st.write("\n")
        
    with features:
        st.header("Özellikler Hakkında")
        sel_col2 , disp_col2 = st.columns(2)
        feature = sel_col2.selectbox('Dağılımını görmek istediğiniz özelliği seçin:',options=list(train.columns)[3:],index=0)
        st.subheader(f"{feature} Dağılımı")
        hist_values = np.histogram(train[feature], bins=5)[0]
        df = pd.DataFrame(hist_values,index=np.arange(0,1.25,0.25))
        st.bar_chart(df)
        st.markdown("* **Test**: Deneme")
        

if __name__ == '__main__':
    train  = pd.read_csv("data/numerai_training_data.csv",nrows=50000)
    stream()
