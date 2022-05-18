import streamlit as st

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
with dataset:
    st.header("Numeraire (NMR) token nedir ?")
    st.write("""Numeraire (NMR) Token: An ERC-20-based token which is used to stake on Numerai. NMR can be earned by competing 
in the Numerai tournament and by making technical contributions to Numerai.""")
