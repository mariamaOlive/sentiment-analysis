import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# Set page title
st.title('Amazon Reviews - Firestick TV')
st.caption("This dashboard was created to display the outcomes of a sentiment \
           analysis study conducted on the Amazon Firestick reviews. The primary \
           objective of the study was to categorize reviews as either positive or \
           negative. Moreover, we aimed to identify the most representative reviews\
           of each category and the most frequently used words in those reviews. \
           To achieve this, all data has been preprocessed beforehand, and classifiers\
           such as Random Forest and BERT have been explored. Some interesting findings can be found bellow.")

@st.cache_data
def load_data():
    df = pd.read_csv("data/reviews_v2.csv")
    df = df[df["reviews"].notna()]
    df["class"] = df["stars"].apply(lambda x : "Positiva" if x >=4 else "Negativa")
    df["dates"] = pd.to_datetime(df["dates"])
    return df

@st.cache_data
def load_pred():
    df = pd.read_csv("data/best_pred.csv")
    df["pred"] = df["pred"].apply(lambda x : "Positiva" if x == 1 else "Negativa")
    df["dates"] = pd.to_datetime(df["dates"])
    return df

@st.cache_data
def load_rep():
    df = pd.read_csv("data/reviews_rep.csv")
    return df

@st.cache_data
def load_metrics():
    df = pd.read_csv("data/metrics.csv").drop("predictions", axis=1)
    return df

@st.cache_data
def load_stopword():
    df = pd.read_csv("portuguese_stopwords.txt", header=None, names=['Words'])
    stopwords = df['Words'].tolist()
    stopwords = [i.replace(' ', '') for i in stopwords] 
    return stopwords


#Loading data
data_load_state = st.text('Loading data...')
data = load_data()
data_pred = load_pred()
data_rep = load_rep()
data_metrics = load_metrics()
pt_stopwords = load_stopword()
data_load_state.text("Data Loaded!")



### Cloud and most representative - Reviews ###
col3, col4 = st.columns(2 , gap="large")

with col3:
    ### Plot Wordcloud ###
    st.header("Wordcloud")
    data_load_state = st.text('Loading data...')
    lista_rev = data["reviews"].tolist()
    big_string = " ".join(lista_rev)
    wordcloud = WordCloud(stopwords=pt_stopwords, background_color="white", max_words=200, contour_width=3, width=800, height=400).generate(big_string)
    st.image(wordcloud.to_array())
    data_load_state.text("")

with col4:
    #Review mais representativos
    st.header("More representative reviews")
    st.subheader("Negative")
    st.markdown('- '+ data_rep["negativo"][0])
    st.markdown('- '+ data_rep["negativo"][1])
    st.subheader("Positive")
    st.markdown('- '+ data_rep["positivo"][0])
    st.markdown('- '+ data_rep["positivo"][1])

st.text("")
st.text("")
st.text("")
### Plot Time Series - Reviews ###
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("Reviews in time - Original (all the data)")
    df_count_reviews = pd.DataFrame({'count' : data.groupby( [ "dates", "class"] ).size()}).reset_index()
    fig = px.line(df_count_reviews, x="dates", y="count", 
                color='class', 
                    color_discrete_map={
                    "Positiva": "#00FF00",
                    "Negativa": "red"
                    },
                    labels={
                            "dates": "Date",
                            "count": "Number of reviews",
                            "class": "Review Type"
                            },
                            title="Reviews in time")
    st.plotly_chart(fig)

with col2:
    st.header("Reviews in time - BERT prediction (test)")
    df_count_reviews = pd.DataFrame({'count' : data_pred.groupby( [ "dates", "pred"] ).size()}).reset_index()
    fig2 = px.line(df_count_reviews, x="dates", y="count", 
                    color='pred', 
                    color_discrete_map={
                    "Positiva": "#00FF00",
                    "Negativa": "red"
                    },
                    labels={
                            "dates": "Date",
                            "count": "Number of reviews",
                            "class": "Review Type"
                            },
                            title="Reviews in time")

    fig2.update_layout(legend_traceorder="reversed")
    st.plotly_chart(fig2)




#Métricas dos classificadores
st.header("Classifiers' Metrics")
st.table(data_metrics)
