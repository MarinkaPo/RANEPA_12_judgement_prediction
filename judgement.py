import streamlit as st
from PIL import Image
# import cloudpickle as cp
# import urllib
# from urllib.request import urlopen

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

import xgboost
import sklearn
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import GridSearchCV
import joblib
from joblib import dump, load
import pickle

# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score
## for data

# import re
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
img_judgement = Image.open('judgement_prediction.jpeg') #
st.image(img_judgement, width=450) #use_column_width='auto'

st.write("""
Приложение *"Supreme Court Judgement Prediction"* демонстрирует, как при помощи методов NLP можно предсказвыть решения суда, 
имея данные о предыдущих делах и их итогах.

\nДанные подготовили сотрудники ЛИА РАНХиГС.
""")
img_pipeline = Image.open('Pipeline_for_Streamlit.png') #
st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн для приложения') #width=450

#-------------------------Project description-------------------------
expander_bar = st.expander("Информация о работе с текстами:")
expander_bar.markdown(
    """
    \nОбработка естественного языка (Natural Language Processing, NLP) — 
    пересечение машинного обучения, нейронных сетей и математической лингвистики, направленное на изучение методов анализа и 
    синтеза естественного языка.
    \n**Используемые библиотеки:** [xgboost](https://xgboost.readthedocs.io/en/stable/), [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [numpy](https://numpy.org/doc/stable/reference/index.html).
    \n**Полезно почитать:** [NLP и LawTech](https://www.law.ox.ac.uk/sites/files/oxlaw/ai_final1097.pdf), 
    [Law and Word Order: NLP in Legal Tech](https://towardsdatascience.com/law-and-word-order-nlp-in-legal-tech-bd14257ebd06), [Stanford CoreNLP](https://cloudacademy.com/blog/natural-language-processing-stanford-corenlp/).

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
st.write('''Рассмотрим пайплайн с моделью XGBClassifier:''')

#-------------------------Выбор слов для предсказания-------------------------
chose_words = st.multiselect(
     'Выберите нужные слова дела',
        ['osha', # для дел, выигранных истцом (==1)
        'segregated',
        'conrail',
        'morrissey',
        'lara',
        'cheney',
        'entergy',
        'isaac',
        'donaldson',
        'papai',
        'mena',
        'barn',
        'donation',
        'aguilar',
        'lecs',
        'robinson',
        'ash',
        'dirk',
        'pg',
        'adarand',
        'ake',
        'generes',
        'butler',
        'hishon',
        'henderson',
        'dickerson',
        'subjectmatter',
        'hawaii',
        
        'daca', # для дел, выигранных ответчиком (==0)
        'map',
        'varsity',
        'hartford',
        'invalidate',
        'kirkpatrick',
        'larceny',
        'iran',
        'vasquezs',
        'whitfield',
        'herrera',
        'bop',
        'fmla',
        'sanchez',
        'fdcpa',
        'export',
        'creation',
        '2015',
        'fiscal',
        'monte',
        'medellin',
        'abramski']
     )
chose_year = st.multiselect('Выберите год', [str(i) for i in range (1955, 2011)])
chose_issue = st.multiselect('Выберите область права',
        ['Economic Activity', 'First Amendment', 'Civil Rights', 'Privacy',
       'Judicial Power', 'Criminal Procedure', 'Due Process',
       'Federalism', 'Federal Taxation', 'Private Action', 'Unions',
       'Attorneys', 'Miscellaneous', 'Interstate Relations'])
st.write('Вы выбрали:', chose_words, chose_year, chose_issue)

if chose_words and chose_year and chose_issue is not None:
    str_words = ' '.join(chose_words)
    example = {'example_fact': str_words,
        'example_year': chose_year,
        'example_issue': chose_issue}
    example_df = pd.DataFrame(data=example, index=[0])
    # example_df

if st.button('Посмотрим предсказания модели'):
    with open("all_sep_words", "rb") as sw:
        all_sep_words = pickle.load(sw)
    df_nl1 = pd.read_csv('df_nl1.csv')

    # фитим векторайзер для слов из дела:
    vectorize=CountVectorizer()
    vectorize.fit(all_sep_words)
    # трансформим:
    count_matrix = vectorize.transform(example_df['example_fact'])
    count_array = count_matrix.toarray()
    data_facts = pd.DataFrame(data=count_array,columns = vectorize.get_feature_names())
    
    # фитим векторайзер для года:
    vectorize_year=CountVectorizer()
    vectorize_year.fit(df_nl1['term'].astype('string'))
    # трансформим:
    count_matrix_year = vectorize_year.transform(example_df['example_year'])
    count_array_year = count_matrix_year.toarray()
    df_year_v2 = pd.DataFrame(data=count_array_year, columns = vectorize_year.get_feature_names())
    
    # фитим векторайзер для сферы права:
    vectorize_issue=CountVectorizer()
    vectorize_issue.fit(df_nl1['issue_area'])
    # трансформим:
    count_matrix_issue = vectorize_issue.transform(example_df['example_issue'])
    count_array_issue = count_matrix_issue.toarray()
    df_issue_v2 = pd.DataFrame(data=count_array_issue, columns = vectorize_issue.get_feature_names())
    
    example_data_final = pd.concat([data_facts, df_year_v2, df_issue_v2],axis=1)

    scaler = joblib.load("StandardScaler.save") 
    example_X_test = scaler.transform(example_data_final)

    # model_XGBC_loaded = pickle.load(urllib.request.urlopen("https://drive.google.com/file/d/1_U1XxZh1W4vjDJFBDQjzltD4muPEO0aQ/view?usp=sharing")) 
    
    with open("XGBClassifier.pkl", "rb") as xgbc:
        model_XGBC_loaded = pickle.load(xgbc)

    # model_XGBC_loaded = pickle.load(open('XGBClassifier.pkl', "rb"))

    # pickle_model = open("XGBClassifier.pkl", "rb")
    # model_XGBC_loaded = pickle.load(pickle_model)

    # model_XGBC_loaded = joblib.load("XGBClassifier.pkl") 
    result = model_XGBC_loaded.predict(example_X_test)
    
    if result[0] == 1:
        st.write('Выиграл истец')
    elif result[0] == 0:
        st.write('Выиграл ответчик')
    else:
        st.write('Нужна корректировка модели')

# st.dataframe(data_final.head(5))
# st.write("Весь размер таблицы: строк:", data_final.shape[0], "столбцов: ", data_final.shape[1])
# st.dataframe(lda_data_train.head(5))
# st.write("Весь размер таблицы: строк:", lda_data_train.shape[0], "столбцов: ", lda_data_train.shape[1])
# st.dataframe(lda_data_test.head(5))
# st.write("Весь размер таблицы: строк:", lda_data_test.shape[0], "столбцов: ", lda_data_test.shape[1])




