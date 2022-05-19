import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache
def get_data():
    train  = pd.read_csv("data/numerai_training_data.csv")
    train["erano"] = train.era.str.slice(3).astype(int)

    return train
def stream():
    train = get_data()
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
        #ERA NO COLUMN
        eras = train.erano

        #FEATURES
        features = [c for c in train if c.startswith("feature")]

        #FEATURE GROUPS
        feature_groups = {
            g: [c for c in train if c.startswith(f"feature_{g}")]
            for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
            }        
        
        st.header("Özellikler Hakkında")
        sel_col2 , disp_col2 = st.columns(2)
        selected_feature = sel_col2.selectbox('Dağılımını görmek istediğiniz özelliği seçin:',options=features,index=0)
        st.subheader(f"{selected_feature} Dağılımı")
        hist_values = np.histogram(train[selected_feature], bins=5)[0]
        df = pd.DataFrame(hist_values,index=np.arange(0,1.25,0.25))
        st.bar_chart(df)

        #CORRELATION
        st.markdown("* **Test**: Deneme")
        st.header('CORRELATION')
        sel_col3 , disp_col3 = st.columns(2)
        selected_feature_for_corr = sel_col3.selectbox('Correlation feature:',options=list(feature_groups.keys()),index=0)
        st.subheader(f"{selected_feature_for_corr} Correlation Table".title())
        corrs = train[feature_groups[f'{selected_feature_for_corr}']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corrs, ax=ax)
        st.write(corrs)
        st.subheader(f"{selected_feature_for_corr} Correlation Heatmap".title())
        st.pyplot(fig)

        #ERAS
        st.subheader("Zamana Bağlı Değişim")
        st.write("Zamanlara Göre Veri Yoğunluğu")
        fig, ax = plt.subplots()
        ax = train.groupby(eras).size().plot()
        st.plotly_chart(fig)

        #predictivity
        corr_with_target = train[features].corrwith(train['target'])
        st.write(pd.DataFrame(corr_with_target.sort_values(ascending=False).head(10),columns=['Correlation With Target']))

    with model_training:
        pass
        

if __name__ == '__main__':
    stream()
