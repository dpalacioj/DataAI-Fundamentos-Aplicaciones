import os
import pandas as pd
import streamlit as st
import pickle
import joblib
from sklearn.pipeline import Pipeline

def get_user_data() -> pd.DataFrame:
    """
    Recoge los datos ingresados por el usuario a través de la interfaz de Streamlit,
    los preprocesa y retorna un DataFrame listo para alimentar el modelo.
    """
    user_data = {}

    # Dividir la pantalla en dos columnas para ingresar datos numéricos
    col_a, col_b = st.columns(2)
    with col_a:
        user_data["age"] = st.number_input(
            label="Edad:", min_value=0, max_value=100, value=20, step=1
        )
        user_data["sibsp"] = st.slider(
            label="Número de hermanos y cónyuges a bordo:",
            min_value=0, max_value=15, value=3, step=1,
        )
    with col_b:
        user_data["fare"] = st.number_input(
            label="Costo del boleto:",
            min_value=0, max_value=300, value=80, step=1,
        )
        user_data["parch"] = st.slider(
            label="Número de padres e hijos a bordo:",
            min_value=0, max_value=15, value=3, step=1,
        )

    # Dividir en tres columnas para seleccionar opciones categóricas
    col1, col2, col3 = st.columns(3)
    with col1:
        user_data["pclass"] = st.radio(
            label="Clase del boleto:", options=["1st", "2nd", "3rd"], horizontal=False
        )
    with col2:
        user_data["sex"] = st.radio(
            label="Sexo:", options=["Woman", "Man"], horizontal=False
        )
    with col3:
        user_data["embarked"] = st.radio(
            label="Puerto de embarque:",
            options=["Cherbourg", "Queenstown", "Southampton"],
            index=1,
        )

    # Convertir el diccionario a DataFrame y transponerlo para tener una fila con todas las variables
    df = pd.DataFrame.from_dict(user_data, orient="index").T

    # Preprocesamiento: mapear los valores de texto a los formatos esperados por el modelo
    df["sex"] = df["sex"].map({"Man": "male", "Woman": "female"})
    df["pclass"] = df["pclass"].map({"1st": 1, "2nd": 2, "3rd": 3})
    df["embarked"] = df["embarked"].map({
        "Cherbourg": "C",
        "Queenstown": "Q",
        "Southampton": "S",
    })

    return df

# @st.cache_resource
# def load_model_pickle(model_file_path: str) -> Pipeline:
#     """
#     Carga un modelo guardado en formato pickle (.pickle).
#     Se usa un spinner para indicar que se está cargando el modelo.
#     """
#     with st.spinner("Cargando modelo..."):
#         with open(model_file_path, "rb") as f:
#             model = pickle.load(f)
#     return model

#@st.cache_resource
def load_model(model_file_path: str) -> Pipeline:
    """
    Carga un modelo guardado en formato joblib (.joblib).
    Se usa un spinner para indicar que se está cargando el modelo.
    """
    with st.spinner("Cargando modelo..."):
        model = joblib.load(model_file_path)
    return model

def main() -> None:
    # Nombre del modelo que vamos a usar

    model_name = "titanic_classification-random_forest-v1.joblib"
    
    # En Google Colab, __file__ no está definido, así que establecemos manualmente la ruta raíz del proyecto:
    try:
        this_file_path = os.path.abspath(__file__)
        project_path = os.path.dirname(os.path.dirname(this_file_path))
    except NameError:
        project_path = "/content/drive/MyDrive/cursos-para-dictar/UDM/06_clase/titanic"
    
    # Construir la ruta relativa al modelo (asumiendo que está en la carpeta "modelos" dentro de 'titanic')
    model_path = os.path.join(project_path, "modelos", model_name)
    
    # Construir la ruta relativa a la imagen del Titanic
    image_path = os.path.join(project_path, "imagenes", "titanic.jpg")
    
    # Verificar que los archivos existen (opcional)
    st.write("Ruta del modelo:", model_path)
    st.write("Existe el modelo?", os.path.exists(model_path))
    st.write("Ruta de la imagen:", image_path)
    st.write("Existe la imagen?", os.path.exists(image_path))
    
    # Mostrar la imagen del Titanic
    st.image(image_path, caption="Esto fue una foto del Titanic")
    
    # Título de la aplicación
    st.header("¿Sobrevivirías al Titanic? 🚢")
    
    # Recoger los datos del usuario
    df_user_data = get_user_data()
    
    # Cargar el modelo
    model = load_model(model_file_path=model_path)
    
    # Realizar la predicción con los datos del usuario
    state = model.predict(df_user_data)[0]
    
    # Definir emojis para la visualización
    emojis = ["😕", "😀"]
    
    st.write("")
    st.title(f"Chance to survive: {emojis[state]}")
    
    if state == 0:
        st.error("¡Mala noticia, amigo! ¡Serás comida de tiburones! 🦈")
    else:
        st.success("¡Felicidades! ¡Puedes estar tranquilo, sobrevivirías al Titanic y ganarías el curso de Analítica de Datos! 🤩")
    
if __name__ == "__main__":
    main()
