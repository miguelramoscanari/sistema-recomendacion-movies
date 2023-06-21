'''
Instalar servidor web
!pip install uvicorn # servidor web

Load servidor, se mantiene escuchando cada cambio
uvicorn main:app --reload

En el browse ser usar
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs

'''

#Importación de las librerías necesarias.
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd

#Le doy a mi FastApi un título, una descripción y una versión.
app = FastAPI(title='Proyecto Individual ML OPS Miguel Ramos',
            description='Partime 01',
            version='1.0.1')

#Primera función donde la API va a tomar mi dataframe para las consultas.
@app.get('/')
async def read_root():
    return {'Mi primera API'}
    
@app.on_event('startup')
async def startup():
    global df
    df = pd.read_csv(r'./Dataset/movies_final.zip', parse_dates=['release_date'])
    recomendacion_previo()

#función para que reconozca mi servidor local
@app.get('/')
async def index():
    return {'API realizada por Miguel Angel Ramos Cañari'}

@app.get('/about/')
async def about():
    return {'Proyecto individual de la cohorte partime 01 de Data Science'}


@app.get('/cantidad_filmaciones_mes/{Mes}')
async def cantidad_filmaciones_mes(Mes: str): 
    # Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron
    # estrenadas en el mes consultado en la totalidad del conjunto de datos.
    # Ejemplo de retorno: X cantidad de peliculas fueron estrenadas en el mes de X
    mes_lista = {'enero': 11, 
                'febrero': 22,
                'marzo': 33,
                'abril': 4,
                'mayo': 5,
                'junio': 6,
                'julio': 7,
                'agosto': 8,
                'setiembre': 9,
                'octubre': 10,
                'noviembre': 11,
                'diciembre': 12}
    # Validando el mes
    Mes = Mes.lower()
    if not Mes in mes_lista:
        return 'Se debe ingresar un mes en idioma español.'
    mes_nro = mes_lista.get(Mes)
    cantidad = len(df[df['release_date'].dt.month == mes_nro])
    
    return {'mes':Mes, 'cantidad':cantidad}    
    
@app.get('/cantidad_filmaciones_dia/{dia}')
async def cantidad_filmaciones_dia(Dia: str):
    # Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que
    # fueron estrenadas en el día consultado en la totalidad del conjunto de datos.
    # Ejemplo de retorno: X cantidad de películas fueron estrenadas en los días X
    dia_lista = {'lunes': 0, 
            'martes': 1,
            'miercoles': 2,
            'jueves': 3,
            'viernes': 4,
            'sabado': 5,
            'domingo': 6}
    
    # Validando el dia
    Dia = Dia.lower()
    if not Dia in dia_lista:
        return 'Se debe ingresar un dia en idioma español.'
    dia_nro = dia_lista.get(Dia)
    cantidad = len(df[df['release_date'].dt.dayofweek == dia_nro])

    return {'dia':Dia, 'cantidad':cantidad}

@app.get('/score_titulo/{titulo}')
async def score_titulo(titulo_de_la_filmación: str):
    # Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
    #Ejemplo de retorno: La película X fue estrenada en el año X con una puntuación/popularidad de X
    
    titulo = titulo_de_la_filmación.lower()
    result = df[df['title'].str.lower() == titulo]
    if len(result) == 0:
        return 'La pelicula no existe'
    
    anio = result['release_year']
    anio = anio.to_list()
    anio = anio[0]
    
    popularidad = result['popularity']
    popularidad = popularidad.to_list()
    popularidad = popularidad[0]
    
    return {'titulo':titulo, 'anio':anio, 'popularidad':popularidad}

@app.get('/votos_titulo/{titulo}')
async def votos_titulo(titulo_de_la_filmación: str):
    # Se ingresa el título de una filmación esperando como respuesta el título, 
    # la cantidad de votos y el valor promedio de las votaciones. La misma variable 
    # deberá contar con al menos 2000 valoraciones, caso contrario, debemos contar
    # con un mensaje avisando que no cumple esta condición y que por ende, 
    # no se devuelve ningun valor.
    # Ejemplo de retorno: La película X fue estrenada en el año X. 
    # La misma cuenta con un total de X valoraciones, con un promedio de X

    titulo = titulo_de_la_filmación.lower()
    result = df[df['title'].str.lower() == titulo]
    if len(result) == 0:
        return 'La pelicula no existe'
    
    vote_count = result['vote_count']
    vote_count = vote_count.to_list()
    vote_count = vote_count[0]
    
    vote_average = result['vote_average']
    vote_average = vote_average.to_list()
    vote_average = vote_average[0]
    
    anio = result['release_year']
    anio = anio.to_list()
    anio = anio[0]
    
    if vote_count < 2000:
        return 'La misma filmacion deberá contar con al menos 2000 valoraciones'
    
    return {'titulo':titulo, 'anio':anio, 'voto_total':vote_count, 'voto_promedio':vote_average}

@app.get('/get_actor/{nombre_actor}')
async def get_actor(nombre_actor:str) :
    # Se ingresa el nombre de un actor que se encuentra dentro de un dataset 
    # debiendo devolver el éxito del mismo medido a través del retorno. 
    # Además, la cantidad de películas que en las que ha realizado y el 
    # promedio de retorno. La definición no deberá considerar directores.
    # Ejemplo de retorno: El actor X ha recibido de X cantidad de filmaciones, 
    # el mismo ha obtenido un retorno de X con un promedio de X por filmación

    actor = nombre_actor.lower()
    # actor = 'Tom Hanks'.lower()
    result = df[df['actor'].str.lower().str.contains(actor, regex=True, na=False)]
    if len(result) == 0:
        return 'El actor no existe'
    
    cantidad_filmaciones = len(result)
    retorno_total = result['revenue'].sum() / result['budget'].sum()
    retorno_promedio = result['return'].mean()

    return {'actor':nombre_actor, 'cantidad_filmaciones':cantidad_filmaciones, 'retorno_total':retorno_total, 'retorno_promedio':retorno_promedio}
    

@app.get('/get_director/{nombre_director}')    
async def get_director(nombre_director: str): 
    # Se ingresa el nombre de un director que se encuentra dentro de un 
    # dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    # Además, deberá devolver el nombre de cada película con la fecha de 
    # lanzamiento, retorno individual, costo y ganancia de la misma.

    director = nombre_director.lower()
    result = df[df['director'].str.lower().str.contains(director, regex=True, na=False)]
    if len(result) == 0:
        return 'El director no existe'

    retorno_total_director = result['revenue'].sum() / result['budget'].sum()
    peliculas = result[['title', 'release_year', 'return', 'budget', 'revenue']].to_dict('records')
    
    return {'director':nombre_director, 'retorno_total_director':retorno_total_director, 'peliculas': peliculas}

@app.get('/recomendacion/{titulo}')    
async def recomendacion(titulo: str):
    # Ingresas un nombre de pelicula y te recomienda las similares en una lista
    try:
        idx = indices[titulo]
    except:
        return 'La pelicula no existe'

    # Obtengo las puntuaciones de similitud por pares de todas las peliculas con esa pelicula
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordene las peliculas segun las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obten las puntuaciones de las 5 peliculas mas similares
    sim_scores = sim_scores[1:6]

    # Obtenga los indices de las peliculas
    movie_indices = [i[0] for i in sim_scores]

    # Devuelve el top 5 de peliculas similares
    peliculas = df_ml['title'].iloc[movie_indices].to_list()
    
    return {'lista recomendada': peliculas}

def recomendacion_previo():
    global cosine_sim, df_ml, indices
    # De las caracteristicas de genero, actor, director y overview
    nro_registros = 5000
    features = ['_genres', 'director', 'actor', 'overview', 'title']
    df_ml = df[features].head(nro_registros)
    df_ml = df_ml.dropna(how='any', subset=features)
    import nltk
    import re

    # stopwords
    nltk.download('stopwords')    


    stemmer = nltk.SnowballStemmer("english")

    from nltk.corpus import stopwords
    import string

    stopword = set(stopwords.words("english"))

    # Definimos función de limpieza
    def clean_data(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text=" ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text=" ".join(text)
        return text    

    for feature in features:
        if feature not in  ['title']:
            df_ml[feature] = df_ml[feature].apply(clean_data)

    # Ahora podemos crear nuestra 'sopa de metadatos', que es una cadena que contiene
    # todos los metadatos que queremos alimentar a nuestro vectorizador
    def create_soup(x):
        return x['_genres'] + ' ' + x['director'] + ' ' + x['actor'] + ' ' + x['overview']        
    
    df_ml['soup'] = df_ml.apply(create_soup, axis=1)
    
    # Import CountVectorizer and create the cuount matrix
    from sklearn. feature_extraction.text import CountVectorizer

    count = CountVectorizer()
    count_matrix = count.fit_transform(df_ml['soup'])

    # Compute the cosine similarity matrix based on the count_matrix
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Restablece el indice de nuestro Dataframe principal y construyo el mapeo inverso como antes
    df_ml = df_ml.reset_index()
    indices = pd.Series(df_ml.index, index=df_ml['title']) 
                        
    