import os # Para manejar archivos y rutas
import pickle # Para guardar y cargar el modelo entrenado, objeto de Python en un archivo binario
from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB # MultinomialNB es un algoritmo de clasificación 
#basado en el teorema de Bayes
MODEL_DIR = "models" # Directorio donde se guardará el modelo entrenado
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl") # Ruta completa del modelo
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl") # Ruta del vectorizador todo va 
#a quedar guardado en la misma carpeta 
ANSWERS_PATH = os.path.join(MODEL_DIR, "answers.pkl") # Ruta de las respuestas

def build_and_train_model(train_pairs):
    # train_pairs es una lista de tuplas (pregunta, respuesta)
    questions = [q for q, _ in train_pairs] # Extraemos las preguntas
    answers = [a for _, a in train_pairs] # Extraemos las respuestas
    vectorizer = CountVectorizer() # Creamos el vectorizador
    x = vectorizer.fit_transform(questions) # Convertimos las preguntas a vectores
    unique_answers = sorted(set(answers)) # Obtenemos las respuestas únicas
    answer_to_label = {a: i for i, a in enumerate(unique_answers)} # Mapeamos respuestas a etiquetas
    y = [answer_to_label[a] for a in answers] # Convertimos respuestas a etiquetas numéricas
    model = MultinomialNB() # Creamos el modelo de Naive Bayes
    model.fit(x, y) # Entrenamos el modelo con los datos
    # Crear carpeta para guardar el modelo si no existe
    os.makedirs(MODEL_DIR, exist_ok=True) #makedirs permite crear archivos, ah[i preguntamos, si no existe creela
    #Guardar los objetos entrenados
    with open(MODEL_PATH, "wb") as f: # Guardamos el modelo entrenado en un archivo binario
        pickle.dump(model, f) # Guardamos el modelo usando pickle 
    with open(VECTORIZER_PATH, "wb") as f: # Guardamos el vectorizador en un archivo binario
        pickle.dump(vectorizer, f) # Guardamos el vectorizador usando pickle
    with open(ANSWERS_PATH, "wb") as f: # Guardamos las respuestas únicas en un archivo binario
        pickle.dump(unique_answers, f) # Guardamos las respuestas usando pickle
    print("👌 Modelo entrenado y guardado exitosamente.") # Mensaje de éxito
    return model, vectorizer, unique_answers # Devolvemos el modelo, vectorizador y respuestas únicas
    #Se construye, se entrena y se crea

def load_model():
    if (
        os.path.exists(MODEL_PATH) 
        and os.path.exists(VECTORIZER_PATH) and os.path.exists(ANSWERS_PATH)
        and os.path.exists(ANSWERS_PATH)
    ): # Verificamos que los archivos del modelo, vectorizador y respuestas existan
        with open(MODEL_PATH, "rb") as f: # Cargamos el modelo desde el archivo binario
            model = pickle.load(f) # Cargamos el modelo usando pickle
        with open(VECTORIZER_PATH, "rb") as f: # Cargamos el vectorizador desde el archivo binario
            vectorizer = pickle.load(f) # Cargamos el vectorizador usando pickle
        with open(ANSWERS_PATH, "rb") as f: # Cargamos las respuestas únicas desde el archivo binario
            unique_answers = pickle.load(f) # Cargamos las respuestas usando pickle
        print("📂 Modelo cargado desde disco.") # Mensaje de éxito
        return model, vectorizer, unique_answers # Devolvemos el modelo, vectorizador y respuestas únicas
    else:
        print("⚠️ No hay modelo guardado. Será necesario entrenarlo primero.") # Mensaje de error
        return None, None, None # Devolvemos None si no se encuentran los archivos
    
def predict_answer(model, vectorizer, unique_answers, user_text):
    x = vectorizer.transform([user_text]) # Convertimos la pregunta a un vector usando el vectorizador
    label = model.predict(x)[0] # Predecimos la etiqueta de la respuesta usando el modelo
    return unique_answers[label] # Devolvemos  la respuesta correspondiente a la etiqueta predicha