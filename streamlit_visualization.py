import pandas as pd
from pandas_datareader import test
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import lightgbm
import pickle

@st.cache
def get_data():
    train  = pd.read_csv("data/numerai_training_data.csv")
    train["erano"] = train.era.str.slice(3).astype(int)

    test = pd.read_csv("data/numerai_tournament_data.csv",nrows=10000)

    return train,test
def stream():
    train,test = get_data()
    header = st.container()
    dataset = st.container()
    features = st.container()
    
    with header:
        st.title("Numerai Veri Seti ile Makine Öğrenmesi Modeli Oluşturma")
        st.write("""Numerai, 2015'te kurulmuştur ve https://numer.ai sitesinde haftalık veri bilimi turnuvaları düzenlemektedir.
    Dünyanın dört bir yanından katılan veri bilimciler şifrelenmiş olan veri setini kullanarak borsaya dair tahminler yürütüp bu
    bu tahminleri sisteme yükleyebilirler. Ve katılımcılar tahminlerinin üzerine bahis oynayarak kazanç sağlayabilirler.
    İyi tahminler daha fazla kripto para kazandırırken, başarısız tahminler ise oynanan paranın kaybolmasına neden olur.""")
        st.header("Numeraire (NMR) token nedir ?")
        st.write("""Numeraire (NMR) Token: Numerai'da bahis oynamak için kullanılan ERC-20 tabanlı kripto para birimidir.. 
        NMR turnuvalara katılarak kazanılabilirken aynı zamanda Numerai'a yapılan teknik katkılarla da kazanılabilir.""")
    
    with dataset:
        st.header("VERİ SETİ HAKKINDA")
        sel_col , disp_col = st.columns(2)
        target = train['target']
        rows = sel_col.slider('Gösterilmesini istediğiniz satır sayısını seçiniz:',min_value = 10, max_value = 100, value=20, step = 10)
        st.write(train.head(rows))
        st.write("* **id**: Hisse için şifrelenmiş isim etiketi") #id: Label for the encrypted stock.
        st.write("""* **feature(özellik)**: Özellikler 5 seviyeye bölünmüştür. Özelliklerin grupları şu şekildedir.: 
“feature_intelligence”, “feature_wisdom”, “feature_charisma”, “feature_dexterity”, “feature_strength”, “feature_constitution”.""")
        st.write("\n")
        
    with features:
        #ERA NO COLUMN
        eras = train.erano

        #FEATURES
        features = [c for c in train if c.startswith("feature")]

        #FEATURE GROUPS
        feature_groups = {
            g: [c for c in train if c.startswith(f"feature_{g}")]
            for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
            }        
        
        st.header("ÖZELLİKLER HAKKINDA")
        sel_col2 , disp_col2 = st.columns(2)
        selected_feature = sel_col2.selectbox('Dağılımını görmek istediğiniz özelliği seçin:',options=features,index=0)
        st.subheader(f"{selected_feature} Dağılımı")
        hist_values = np.histogram(train[selected_feature], bins=5)[0]
        df = pd.DataFrame(hist_values,index=np.arange(0,1.25,0.25))
        st.bar_chart(df)

        #CORRELATION
        st.header('CORRELATION')
        sel_col3 , disp_col3 = st.columns(2)
        selected_feature_for_corr = sel_col3.selectbox('Correlation için bir özellik seçin:',options=list(feature_groups.keys()),index=0)
        st.subheader(f"{selected_feature_for_corr} Correlation Table".title())
        corrs = train[feature_groups[f'{selected_feature_for_corr}']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corrs, ax=ax)
        st.write(corrs)
        st.subheader(f"{selected_feature_for_corr} Correlation Heatmap".title())
        st.pyplot(fig)

        #ERAS
        st.subheader("ZAMANA BAĞLI DEĞİŞİM")
        st.write("Zamanlara Göre Veri Yoğunluğu")
        fig, ax = plt.subplots()
        ax = train.groupby(eras).size().plot()
        st.plotly_chart(fig)

        #predictivity
        st.subheader("ÖZELLİKLERİN HEDEFLE OLAN BAĞINTISI")
        corr_with_target = train[features].corrwith(train['target'])
        st.write(pd.DataFrame(corr_with_target.sort_values(ascending=False).head(10),columns=['Correlation With Target']))

        

if __name__ == '__main__':
    stream()
