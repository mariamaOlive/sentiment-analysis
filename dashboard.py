import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# Set page title
st.title('Reviews da Amazon - Firestick TV')

@st.cache
def load_data():
    df = pd.read_csv("data/reviews_v2.csv")
    df = df[df["reviews"].notna()]
    df["class"] = df["stars"].apply(lambda x : "Positiva" if x >=4 else "Negativa")
    df["dates"] = pd.to_datetime(df["dates"])
    return df

@st.cache
def load_pred():
    df = pd.read_csv("data/best_pred.csv")
    df["pred"] = df["pred"].apply(lambda x : "Positiva" if x == 1 else "Negativa")
    df["dates"] = pd.to_datetime(df["dates"])
    return df

@st.cache
def load_rep():
    df = pd.read_csv("data/reviews_rep.csv")
    df
    return df

@st.cache
def load_metrics():
    df = pd.read_csv("data/metrics.csv").drop("predictions", axis=1)
    return df

#Loading data
data_load_state = st.text('Loading data...')
data = load_data()
data_pred = load_pred()
data_rep = load_rep()
data_metrics = load_metrics()
data_load_state.text("Done! (using st.cache)")

# st.subheader('Raw data')
# st.write(data)

col1, col2 = st.columns(2)

### Plot Time Series - Reviews ###
col1.header("Reviews no tempo - Original (todos os dados)")
df_count_reviews = pd.DataFrame({'count' : data.groupby( [ "dates", "class"] ).size()}).reset_index()
fig = px.line(df_count_reviews, x="dates", y="count", 
              color='class', 
                color_discrete_map={
                 "Positiva": "#00FF00",
                 "Negativa": "red"
                },
                labels={
                        "dates": "Data",
                        "count": "Número de reviews",
                        "class": "Tipo de review"
                        },
                        title="Reviews no tempo")
col1.plotly_chart(fig)

col2.header("Reviews no tempo - Predições do BERT (conjunto de teste)")
df_count_reviews = pd.DataFrame({'count' : data_pred.groupby( [ "dates", "pred"] ).size()}).reset_index()
fig2 = px.line(df_count_reviews, x="dates", y="count", 
                color='pred', 
                color_discrete_map={
                 "Positiva": "#00FF00",
                 "Negativa": "red"
                },
                labels={
                        "dates": "Data",
                        "count": "Número de reviews",
                        "pred": "Tipo de review"
                        },
                        title="Reviews no tempo")

fig2.update_layout(legend_traceorder="reversed")
col2.plotly_chart(fig2)

col3, col4 = st.columns(2)

### Plot Wordcloud ###
col3.header("Wordcloud")
data_load_state = st.text('Loading data...')
lista_rev = data["reviews"].tolist()
big_string = (" ").join(lista_rev)
stopwords = stopwords.words('portuguese')
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=200, contour_width=3, width=800, height=400).generate(big_string)
fig3 = plt.figure(figsize=[20,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
col3.pyplot(fig3)
data_load_state.text("")


#Review mais representativos
col4.header("Reviews mais representativos")
col4.subheader("Negativos")
col4.markdown('- '+ data_rep["negativo"][0])
col4.markdown('- '+ data_rep["negativo"][1])
col4.subheader("Positivos")
col4.markdown('- '+ data_rep["positivo"][0])
col4.markdown('- '+ data_rep["positivo"][1])

#Métricas dos classificadores
st.header("Métricas dos classificadores")
st.table(data_metrics)
