import streamlit as st
from PIL import Image

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import joblib
from joblib import dump, load

# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score
## for data

import re
# import nltk # Natural Language Toolkit - пакет библиотек и программ для символьной и статистической обработки естественного языка
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# import wordcloud

# import gensim.downloader as gensim_api
# import gensim

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# ---------------------Header---------------------
st.markdown('''<h1 style='text-align: right; color: green;'
            >Supreme Court Judgement Prediction</h1>''', 
            unsafe_allow_html=True)
img = Image.open('judgement_prediction.jpeg') #
st.image(img, width=450) #use_column_width='auto'

st.write("""
Приложение *"Supreme Court Judgement Prediction"* демонстрирует, как при помощи методов NLP можно предсказвыть решения суда, 
имея данные о предыдущих делах и их итогах.

\nДанные подготовил ...
""")
#-------------------------Project description-------------------------
expander_bar = st.expander("Информация о работе с текстами:")
expander_bar.markdown(
    """
    \nНаписать про NLP и т.д.
    \n**Используемые библиотеки:** [...](ДОБАВИТЬ ССЫЛКУ), [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [numpy](https://numpy.org/doc/stable/reference/index.html).
    \n**Полезно почитать:** [Ссылка 1](ДОБАВИТЬ ССЫЛКУ), 
    [Ссылка 2](ДОБАВИТЬ ССЫЛКУ)), [Ссылка 3](ДОБАВИТЬ ССЫЛКУ)).

    """)

# ---------------------Reading CSV---------------------
df = pd.read_csv('justice.csv', delimiter=',', encoding = "utf8")

# lda_data_train = pd.read_csv('lda_data_train.csv')
# lda_data_test = pd.read_csv('lda_data_test.csv')

st.markdown('''<h1 style='text-align: center; color: black;'> Блок 1 </h1>''', 
            unsafe_allow_html=True)
st.header('Смотрим датасет')
st.dataframe(df.head(5))
st.write("Весь размер таблицы: строк:", df.shape[0], "столбцов: ", df.shape[1])
expander_bar = st.expander('Информация о датасете')
expander_bar.markdown(f'''Перед нами датасет, содержащий информацию о решениях суда в США по {df.shape[0]} делам за период с 1955 по 2020 года. 
\n**Данные распределены по {df.shape[1]} колонкам:**
\n'Unnamed: 0' - номер
\n'ID' - ID дела
\n'name' - лица, участвующие
\n'href' - ссылка
\n'docket' - маркировка дела
\n'term' - год (с 1955 по 2020)
\n'first_party' - истец
\n'second_party' - ответчик
\n'facts' - данные дела
\n'facts_len' - количество символов
\n'majority_vote' - большинство голосов
\n'minority_vote' - меньшинство голосов
\n'first_party_winner' - выиграл ли истец (если False - то выиграл ответчик)
\n'decision_type' - вид принятия решения
\n'disposition' - окончательное решение суда по обвинению
\n'issue_area' - вид дела: гражданское, уголовное и др.
'''
)
st.header('Препроцессинг текста')
st.write("""Часть данных можно удалить, а наиболее ценные фичи превратим в числовую матрицу, для передачи в модель.
\n**Были проведены:**
\nОчистка данных от лишних символов:""")
st.code('''
df_nlp1 = pd.DataFrame(df_nlp, columns=['facts'])
df_nlp1['facts'] = df_nlp1['facts'].str.replace(r'<[^<>]*>', '', regex=True)
''')
st.write("""\nТокенизация материалов дела:""")
st.code('''corpus = df_nlp1["facts"]
lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))''')
st.write("""\nПрепроцессинг текста:""")
st.code('''
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## очистка текста (переводим в нижний регистра, удаляем знаки препинания, лишние пробелы в начале и в конце)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## "ручная" токенизация (разбиваем предложения на слова: из строки в список):
    lst_text = text.split()    
    if lst_stopwords is not None: ## убираем Stopwords
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Стемминг (убираем окончания, приставки и т.д., приводя слово к его основе)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Лемматизация (приводим слово к его нормальной (словарной) форме)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## возвращаем обратно в строку:
    text = " ".join(lst_text)
    return text''')
st.write('''One_hot_encoding категориальных колонок:''')
st.code('''
df_cat1 = pd.get_dummies(df_cat['decision_type'])
df_cat2 = pd.get_dummies(df_cat['disposition'])
df_cat3=pd.concat([df_cat2,df_cat1],axis=1,join='inner')
df_cat3=pd.concat([df_cat3,df_nl1['first_party_winner']],axis=1,join='inner')''')
st.write('''Векторизация данных:''')
st.code('''
vectorize=CountVectorizer()
count_matrix = vectorize.fit_transform(df_nl1['facts_clean'])
count_array = count_matrix.toarray()''')
# data_final = pd.read_csv('data_final.csv')
st.write(f'''**Таким образом, из таблицы размером {df.shape[0]}х{df.shape[1]} 
мы получили таблицу размером 3098х20270 - так называемый bag_of_words:**''')
b_o_w = Image.open('bag_of_words.png') #
st.image(b_o_w, use_column_width='auto') #width=400

st.header('Разделим на train (X и y) test (X и y) ')
st.code('''X_train, X_test, y_train, y_test = train_test_split(data_final.drop(columns=['first_party_winner']), 
                                                    data_final['first_party_winner'], 
                                                    test_size=0.3,
                                                    random_state=10)''')

st.markdown('''<h1 style='text-align: center; color: black;'> Блок 2 </h1>''', 
            unsafe_allow_html=True)
st.header('Работа с моделью')
st.write('''Рассмотрим пайплайн с моделью RandomForestClassifier():''')

#-------------------------Выбор слов для предсказания-------------------------
options = st.multiselect(
     'Выберите нужные слова',
     ['younger',
        'wheat',
        'panikos',
        'Oxford',
        'directorate',
        'kaley',
        'beilan',
        'depot',
        'wreck',
        'rabkin',
        'annotation',
        'dead',
        'give up',
        'decrease',
        'century',
        'miracle',
        'scammer',
        'Appendix']
     )
st.write('Вы выбрали:', options)

if options is not None:
    my_str = ' '.join(options)
    example_text = {'example_fact': my_str}
    example_fact_clean = pd.DataFrame(data=example_text, index=[0])
    example_X_test = example_fact_clean['example_fact']

if st.button('Посмотрим предсказания модели'):
    pipeline_RFC = joblib.load('pipeline_RFC_2.pkl') # pipe1.pkl
    ex_pred_RFC = pipeline_RFC.predict(example_X_test)
    if ex_pred_RFC[0] == 1:
        st.write('Выиграл истец')
    else:
        st.write('Выиграл ответчик')

# st.dataframe(data_final.head(5))
# st.write("Весь размер таблицы: строк:", data_final.shape[0], "столбцов: ", data_final.shape[1])
# st.dataframe(lda_data_train.head(5))
# st.write("Весь размер таблицы: строк:", lda_data_train.shape[0], "столбцов: ", lda_data_train.shape[1])
# st.dataframe(lda_data_test.head(5))
# st.write("Весь размер таблицы: строк:", lda_data_test.shape[0], "столбцов: ", lda_data_test.shape[1])




