# Sistema de Recomendación de Peliculas
Sistema de Recomendación, usando Machine Learning, permite recomendar películas basandose en películas similiares, en base a un score de similaridad.

# Contenido
- "Dataset/movies_final.zip": fuente de datos
- "ETL_Miguel_Ramos.ipynb": notebook proceso ETL
- "EDA_ML_Miguel_Ramos.pynb": notebook proceso EDA y ML
- "main.py": API funciones
- "index.html": interfaz web amigable para usar el API
# API
Se creo siete funciones de consulta, usando FastApi y para deploy Render:
![Image text](https://github.com/miguelramoscanari/sistema-recomendacion-movies/blob/main/web_api.png)

# Proceso ETL
Lo relevenante de este proceso fue convertir columna de lista de diccionarios a string, de las siguientes:
- belongs_to_collection : Un diccionario que indica a que franquicia o serie de películas pertenece la película
- budget : El presupuesto de la película, en dólares
- genres : Un diccionario que indica todos los géneros asociados a la película
- production_companies : Lista con las compañias productoras asociadas a la película
- production_countries : Lista con los países donde se produjo la película
- spoken_languages : Lista con los idiomas que se hablan en la pelicula
- cast : actores de la pelicula
- crew : directores de la pelicula

# Proceso EDA
Tareas realizadas en EDA
- Revisando valores nulos, imputando valores nulos y limpiando datos
- Analisis de contenido por genero
- Analisis de contenido por año
- Analisis de contenido por mes
- Analisis de duracion de movies
- Paises que mas peliculas realizan

# Proceso ML
Para este sistema de recomendacion, se uso Contenido basado en filtro. 
El contenido de la pelicula(reparto, descripcion, director, genero) se utiliza para encontrar similitud con otra peliculas. 
Entonces se recomienda las peliculas que tienen mas probabilidadas de ser similares
