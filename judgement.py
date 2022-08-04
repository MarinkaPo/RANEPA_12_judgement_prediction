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
import plotly.express as px

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

# https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction

# ---------------------Header---------------------
st.markdown('''<h1 style='text-align: center; color: green;'
            >Предсказание решений суда</h1>''', 
            unsafe_allow_html=True)
st.markdown('''<h3 style='text-align: center; color: #6a5750;'
            >Supreme Court Judgement Prediction</h3>''', 
            unsafe_allow_html=True)

# st.image('judgement_prediction.jpeg', width=650) 

col1, col2, col3 = st.columns([1,2,1])
with col1:
	st.write("")
with col2:
	st.image('judgement_prediction.jpeg', use_column_width='auto')  # uplift_header_pic.jpg Uplift_modeling.jpg
with col3:
	st.write("")            
# img_judgement = Image.open('judgement_prediction.jpeg') #
# st.image(img_judgement, width=450) #use_column_width='auto'

st.write("""
Приложение *"Supreme Court Judgement Prediction"* демонстрирует, как при помощи методов NLP можно предсказвыть решения суда, 
имея данные о предыдущих делах и их итогах.""")

st.markdown('''<h2 style='text-align: left; color: black;'
            >Цель и задачи:</h2>''', 
            unsafe_allow_html=True)
st.write(""" \nВнимание судьи часто распыляется на множество бесспорных дел и рутинных процедур. В итоге на сложные дела ресурса внимания и времени может не хватить. 
Применение искуственного интеллекта в судебной практике может помочь избавить судей и граждан от рутины, сэкономить время, уменьшить ошибки и необъективность. Он широко применяется в Китае, США, Франции и других государствах. 
 
\n**Цель данной лабораторной работы** - ознакомить студентов юридических специальностей с возможностью применения искуственного интеллекта для прогнозирования исхода судебных дел.  
\n**Задачи:**
\n**1)** Ознакомиться с теоритическими аспектами применения машинного обучения в юриспруденции;
\n**2)** Использовать инструменты графического анализа данных для первичного изучения датасета; 
\n**3)** Попробовать применять прогнозную модель для разных вариантов судебных дел: выполняя задания лабораторной работы, выявить закономерности, которые имеют наибольшее влияние на решение предективной модели.

\n*Данные подготовили сотрудники ЛИА РАНХиГС.*
""")
#-------------------------Pipeline description-------------------------
st.markdown('''<h2 style='text-align: left; color: black;'
            >Пайплайн лабораторной работы:</h2>''', unsafe_allow_html=True)
img_pipeline = Image.open('Pipeline_for_Streamlit.png') #
st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн для приложения') #width=450


pipeline_bar = st.expander("Этапы работы:")
pipeline_bar.markdown(
    """
    \n*(зелёным обозначены этапы, работа с которыми доступна студенту, красным - этапы, доступные для корректировки сотрудникам ЛИА)*
    \n**1. Сбор данных:**
    \nДатасет был взять из [](https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction)
    Набор данных содержит 3304 дела Верховного суда США с 1955 по 2021 год. Каждое дело имеет идентификаторы дела, а также факты дела и результат решения.
    Целевая переменная - это решение по делу: выиграл истец или же ответчик.
    \n**2. Предобработка данных:**
    \nСюда входят:
    \n* Нормализация
    \nЭто перевод всех букв в тексте в нижний регистр, удаление знаков пунктуации,  чисел, пробельных символов
    \nИспользуются методы языка [Python](https://www.python.org/), [регулярные выражения языка Python](https://docs.python.org/3/library/re.html)
    \n* Токенизация
    \nпроцесс разделения предложений на компоненты: слова или словосочетания
    \nС использованием [библиотеки nltk](https://www.nltk.org/)
    \n* Удаление стоп слов
    \nК ним относятся артикли, междометия, союзы и т.д., - т.к. они не несут смысловой нагрузки
    \nИспользуется [библиотека nltk](https://www.nltk.org/)
    \n* Стемминг/лемматизация 
    \nЭто приведение слова к нормальной словарной форме
    \nПрименяется [библиотека nltk](https://www.nltk.org/)
    \n* Векторизация
    \nКонвертартация текста в наборы цифр (векторы) 
    \nПрименяется [библиотека scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).
    \n**3. Выбор baseline-модели, её обучение и валидация:**
    \nБыли использованы [логистическая регрессия](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [метод ближайших соседей](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) и [случайный лес](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
    \n**4. Выбор более сложной модели, её обучение и валидация:**
    \nСамые хорошие результаты показала модель машинного обучения [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html);
    \n**5. Сравнение результатов**:
    \nВыполнялось сотрудником ЛИА
    \n**6. Файнтюнинг (настройка параметров) лучшей модели:**
    \nВыполнялся сотрудником ЛИА
    \n**7. Оформление микросервиса Streamlit, выгрузка на сервер:**
    \nПроводится сотрудником лаборатории, используется студентами РАНХиГС.
    """)

info_bar = st.expander("Информация о применении искуственного интеллекта в судебной практике:")
info_bar.markdown(
    """
    \nРазвитие цифровых технологий в эпоху информационного общества и больших данных доказало перспективу внедрения искусственного интеллекта в суде. Стало очевидным, искусственный интеллект – это наше настоящее, а не будущее, как мы еще недавно утверждали. 
    В последнее время искусственный интеллект используется во многих областях, и правовая система не является исключением. 
    \nПоследние достижения в *NLP (Natural Language Processing, обработка естественного языка - пересечение машинного обучения, нейронных сетей и математической лингвистики, направленное на изучение методов анализа и синтеза естественного языка)* предоставляют нам инструменты 
    для построения прогностических моделей, которые можно использовать для выявления закономерностей, влияющих на судебные решения. Используя передовые алгоритмы NLP для анализа предыдущих судебных дел, обученные модели могут 
    прогнозировать и классифицировать решение суда с учетом фактов дела от истца и ответчика в текстовом формате; другими словами, модель подражает деятельности судьи, вынося окончательный вердикт.
    \nВспоминая предложения руководства Верховного Cуда РФ и Совета судей России относительно постепенного [внедрения в суде «слабого искусственного интеллекта»](http://www.ssrf.ru/news/lienta-novostiei/36912), способного решать узкоспециализированные задачи,
    можно предположить следующие этапы его внедрения в систему отечественных судов:
    \n* краткосрочная перспектива: ИИ как ассистент судьи-человека по ряду вопросов делопроизводства и при рассмотрении дела по существу;
    \n* среднесрочная перспектива (5-10 лет): ИИ как судья-компаньон судьи-человека, в том числе по вопросу оценки ряда доказательств;
    \n* долгосрочная перспектива: возможная замена судьи-человека ИИ по отдельным функциям судьи-человека при осуществлении правосудия.
    \nЕстественно, технология работы судебного искусственного интеллекта должна быть открыта, достоверна и прозрачна для всех граждан, хозяйствующих субъектов и общества в целом. Такой подход обеспечит доверие общества к суду и внедряемым в его работу современным 
    информационным технологиям: искусственному интеллекту и облачным вычислениям.
    \n**Используемые библиотеки:** [xgboost](https://xgboost.readthedocs.io/en/stable/), [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [numpy](https://numpy.org/doc/stable/reference/index.html).
    \n**Полезно почитать:** [NLP и LawTech](https://www.law.ox.ac.uk/sites/files/oxlaw/ai_final1097.pdf), 
    [Law and Word Order: NLP in Legal Tech](https://towardsdatascience.com/law-and-word-order-nlp-in-legal-tech-bd14257ebd06), [Stanford CoreNLP](https://cloudacademy.com/blog/natural-language-processing-stanford-corenlp/), 
    [Перспективы использования ИИ в российской судебной системе](https://pravo.ru/opinion/232129/), [Мировые подходы к применению ИИ в судах](https://www.ng.ru/ideas/2021-04-07/7_8122_ai.html).

    """)    
# ---------------------Reading CSV---------------------
df = pd.read_csv('justice.csv', delimiter=',', encoding = "utf8")

# lda_data_train = pd.read_csv('lda_data_train.csv')
# lda_data_test = pd.read_csv('lda_data_test.csv')

#-------------------------------------БЛОК 1: анализ данных-------------------------------------------
st.markdown('''<h1 style='text-align: center; color: black;'> Блок 1: анализ данных </h1>''', 
            unsafe_allow_html=True)


# st.header('Смотрим датасет')

# expander_bar = st.expander('Информация о датасете')
st.markdown(f'''### Информация о датасете 
\nПеред нами датасет, содержащий информацию о решениях суда в США по {df.shape[0]} делам за период с 1955 по 2020 года.''')
st.dataframe(df.head(5))
st.write("Весь размер таблицы: строк:", df.shape[0], "столбцов: ", df.shape[1])
st.markdown(f'''
\n**Данные распределены по {df.shape[1]} колонкам:**
\n* 'Unnamed: 0' - номер
\n* 'ID' - ID дела
\n* 'name' - лица, участвующие
\n* 'href' - ссылка
\n* 'docket' - маркировка дела
\n* 'term' - год (с 1955 по 2020)
\n* 'first_party' - истец
\n* 'second_party' - ответчик
\n* 'facts' - данные дела
\n* 'facts_len' - количество символов
\n* 'majority_vote' - большинство голосов
\n* 'minority_vote' - меньшинство голосов
\n* 'first_party_winner' - *целевая переменная*: выиграл ли истец (если False - то выиграл ответчик)
\n* 'decision_type' - вид принятия решения
\n* 'disposition' - окончательное решение суда по обвинению
\n* 'issue_area' - вид дела: гражданское, уголовное и др.
''')

#-----------------HistPlot--------------------
with st.form(key='hist_filter'):
  st.markdown('''### Задание к Блоку 1:
  \n**С помощью гистограммы *HistPlot* (график, показывающий распредление признаков) ответьте наследующие вопросы:**
  \n1. В каком году проходили большинство дел из нашего набора данных?
  \n2. За какой год данных меньше всего?
  \n3. Определите топ-3 дел по области права.
  \n4. Какая область права представлена меньше всего в данных? Сколько по ней прошло дел?
  \n5. Кто чаще выигрывал - истец или ответчик?
  
  \n*Для построения графика выберите один нужный признак и нажмите кнопку "Построить гистограмму по признаку"*. 
  \n*График интерактивный: можно увеличить интересующую область, а двойной клик возвращает к исходному гарфику.*''')
  hist_filter = st.multiselect('Выберите один признак:', ['Год дела', 'Область права', 'Выиграл ли истец'])  # ['term', 'issue_area', 'first_party_winner']  ['год дела', 'область права', 'итог дела']
  if st.form_submit_button('Построить гистограмму по признаку'):
    if not hist_filter:
        st.write('*Признак для посторения гистограммы не выбран*')
    else:
        if hist_filter[0] == 'Год дела':
            hist_filter[0] = 'term'
        elif hist_filter[0] == 'Область права':
            hist_filter[0] = 'issue_area'
        else:
            hist_filter[0] = 'first_party_winner' 

        fig = px.histogram(df[hist_filter[0]],
                           df[hist_filter[0]],
                           title='График распределения данных по выбранному признаку',
                           color=hist_filter[0],
                           labels = {'term': 'Год дела:',
                           'issue_area': 'Область права:',
                           'first_party_winner' :'Выиграл ли истец:'})

        fig.update_xaxes(title='Выбранный признак',
                        automargin = True,
                        tickangle =270
                        # categoryorder='total descending'
                        )

        fig.update_yaxes(
            title='Количество дел'
        )

        # fig.update_layout(
        #     showlegend=True,
        #     legend_orientation="h",
        #     legend=dict(x=.66, y=.99, title='Новый клиент'),
        #     margin=dict(l=20, r=10, t=80, b=10),
        #     hovermode="x",
        #     bargap=0.2
        # )

        # fig.update_traces(hovertemplate="Количество клиентов: %{y}")
        st.plotly_chart(fig, use_container_width=True)    

        # fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(20,10))
        # plt.ticklabel_format(style='plain')   
        # ax = sns.histplot(data = df, x = df[hist_filter[0]], kde = True)
        # plt.xticks(rotation=90)
        # # sns.set_theme(style='whitegrid', palette="bright")
        # # sns.color_palette("husl", 9)
        # # sns.color_palette("husl", 8)
        # st.pyplot(fig,style='whitegrid', palette="pastel")

#--------------------------------------------------------------------------------
st.markdown(f'''### Препроцессинг текста
\n ##### *Данный раздел является ознакомительным*''')
st.write("""Целью предобработки текстов является создание т.н. корпуса - подобранной и обработанной по определённым правилам совокупности текстов, используемых в качестве базы для работы, и, 
затем, перевод этого корпуса в формат удобный для дальнейшей работы модели или нейронной сети.
Общий план предобработки текста, сводится к следующим этапам:
    \n• **Нормализация** (перевод всех букв в тексте в нижний регистр, удаление знаков пунктуации,  чисел, пробельных символов). Обычно реализуется методами языка Python;
    \n• **Токенизация** (процесс разделения предложений на компоненты: слова или словосочетания). Обычно реализуется специальными библиотеками (напрмер, [nltk](https://www.nltk.org/)) или методами языка Python; 
    \n• **Удаление стоп слов**. Это такие слова, которые будут мешать обучению модели. К ним относятся артикли, междометия, союзы и т.д., - т.к. они не несут смысловой нагрузки. При этом надо понимать, 
    что не существует универсального списка стоп-слов, все зависит от конкретного случая, и к базовым спискам стоп-слов из библиотек можно добавлять свои; 
    \n• **Стемминг/лемматизация**. Данный этап преследуют цель привести все встречающиеся словоформы к одной, нормальной словарной форме. Стемминг – это грубый эвристический процесс, который отрезает 
    «лишнее» от корня слов, часто это приводит к потере словообразовательных суффиксов. Чаще используется для английских текстов. Лемматизация – это более тонкий процесс, который использует словарь и морфологический анализ, 
    чтобы в итоге привести слово к его канонической форме (лемме). Она чаще применяется для русского языка; 
    \n• **Векторизация**. Алгоритмы машинного обучения не могут напрямую работать с сырым текстом, поэтому необходимо конвертировать текст в наборы цифр (векторы). Для этого можно использовать, например, [CountVectorizer из библиотеки scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)""")
# \n**Были проведены:**
# \nОчистка данных от лишних символов:
# st.code('''
# df_nlp1 = pd.DataFrame(df_nlp, columns=['facts'])
# df_nlp1['facts'] = df_nlp1['facts'].str.replace(r'<[^<>]*>', '', regex=True)
# ''')
# st.write("""\nТокенизация материалов дела:""")
# st.code('''corpus = df_nlp1["facts"]
# lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))''')
# st.write("""\nПрепроцессинг текста:""")
# st.code('''
# def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
#     ## очистка текста (переводим в нижний регистра, удаляем знаки препинания, лишние пробелы в начале и в конце)
#     text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
#     ## "ручная" токенизация (разбиваем предложения на слова: из строки в список):
#     lst_text = text.split()    
#     if lst_stopwords is not None: ## убираем Stopwords
#         lst_text = [word for word in lst_text if word not in 
#                     lst_stopwords]
                
#     ## Стемминг (убираем окончания, приставки и т.д., приводя слово к его основе)
#     if flg_stemm == True:
#         ps = nltk.stem.porter.PorterStemmer()
#         lst_text = [ps.stem(word) for word in lst_text]
                
#     ## Лемматизация (приводим слово к его нормальной (словарной) форме)
#     if flg_lemm == True:
#         lem = nltk.stem.wordnet.WordNetLemmatizer()
#         lst_text = [lem.lemmatize(word) for word in lst_text]
            
#     ## возвращаем обратно в строку:
#     text = " ".join(lst_text)
#     return text''')
# st.write('''One_hot_encoding категориальных колонок:''')
# st.code('''
# df_cat1 = pd.get_dummies(df_cat['decision_type'])
# df_cat2 = pd.get_dummies(df_cat['disposition'])
# df_cat3=pd.concat([df_cat2,df_cat1],axis=1,join='inner')
# df_cat3=pd.concat([df_cat3,df_nl1['first_party_winner']],axis=1,join='inner')''')
# st.write('''Векторизация данных:''')
# st.code('''
# vectorize=CountVectorizer()
# count_matrix = vectorize.fit_transform(df_nl1['facts_clean'])
# count_array = count_matrix.toarray()''')
# data_final = pd.read_csv('data_final.csv')
st.markdown(f''' ##### *Таким образом, после всех преобразований текста*, 
\nиз таблицы со словами, размером {df.shape[0]} строк х {df.shape[1]} колонок мы получили 
\nтаблицу размером 3098 строк х 20270 колонок - так называемый *bag_of_words*:''')
b_o_w = Image.open('bag_of_words.png') #
st.image(b_o_w, use_column_width='auto') #width=400

# st.header('Разделим на train (X и y) test (X и y) ')
# st.code('''X_train, X_test, y_train, y_test = train_test_split(data_final.drop(columns=['first_party_winner']), 
#                                                     data_final['first_party_winner'], 
#                                                     test_size=0.3,
#                                                     random_state=10)''')


#-------------------------------------БЛОК 2: работа с моделью-------------------------------------------
st.markdown('''<h1 style='text-align: center; color: black;'> Блок 2: работа с моделью </h1>''', 
                unsafe_allow_html=True)
with st.form(key='model_filter'):
    st.markdown('''### Задание к Блоку 2:
    \n**Используя модель машинного обучения, выполните следующие задания:**
    \n1. Меняя слова из материала дела, год и область права, попробуйте получить оба возможных исхода судебного разбирательства: **победу истца** или **победу ответчика**;
    \n2. При одинаковых словах материала дела и области права, как влияет год на предсказние модели?
    \n3. При одинаковых словах материала дела и одинаковом годе, как влияет выбор области права на итоговое решение модели?
    \n4. Сделайте вывод по лабораторной работе.
    
    \n*Для использования модели вначале выберете слова для судебного дела, затем год и область права. После этого нажмите на кнопку "Посмотреть предсказания модели"*. 
    \n*Используется модель машинного обучения [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html).*''')

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
            #'2015',
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
    # st.write('Вы выбрали:', chose_words, chose_year, chose_issue)

    if chose_words and chose_year and chose_issue is not None:
        str_words = ' '.join(chose_words)
        example = {'example_fact': str_words,
            'example_year': chose_year,
            'example_issue': chose_issue}
        example_df = pd.DataFrame(data=example, index=[0])
        # example_df

    if st.form_submit_button('Посмотреть предсказания модели'):
        try:
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
                st.markdown('''<h3 style='text-align: left; color: #008000;'>Выиграл истец</h3>''',unsafe_allow_html=True)
            elif result[0] == 0:
                st.markdown('''<h3 style='text-align: left; color: #322a35;'>Выиграл ответчик</h3>''',unsafe_allow_html=True)
            else:            
                st.write('**Нужна корректировка модели**')
        except:
            st.write('*Вы выбрали не все характеристики материалов дела*')

# st.dataframe(data_final.head(5))
# st.write("Весь размер таблицы: строк:", data_final.shape[0], "столбцов: ", data_final.shape[1])
# st.dataframe(lda_data_train.head(5))
# st.write("Весь размер таблицы: строк:", lda_data_train.shape[0], "столбцов: ", lda_data_train.shape[1])
# st.dataframe(lda_data_test.head(5))
# st.write("Весь размер таблицы: строк:", lda_data_test.shape[0], "столбцов: ", lda_data_test.shape[1])




