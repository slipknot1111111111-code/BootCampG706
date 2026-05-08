from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB
"""
MultinomialNB modelo de inteligencia
 artificial que aprende relaciones entre texto y respuestas
"""
#============================================================
# Función build_and_train_model
#============================================================
def build_and_train_model(train_pairs):
  # train_pairs: Lista de pares (pregunta, respuesta)
  # Ejemplo [("Hola","!Hola¡"),("adiós","!Hasta Luego¡")]

  # separamos las preguntas y respuestas en dos listas
  questions = [q for q, _ in train_pairs] # Lista de preguntas
  answers = [a for _, a in train_pairs] # Lista de respuestas
  # creamos el vectorizador, que traducira el texto a números
  vectorizer = CountVectorizer()
  #Entrenamos el vectorizados con las preguntas y las
  # convertimos en números
  x = vectorizer.fit_transform(questions)
  # Obtenemos una lista de respuestas únicas (sin repetir)
  unique_answers = sorted(set(answers))
  # creamos una diccionario que asigne un número a cada respuesta
  # Ejemplo :{"!Hola¡":0, "!Hasta Luedo¡",1}
  answer_to_label = {a: i for i, a in enumerate(unique_answers)}
  #creamos una lista con las etiqutas númericas de las respuestas
  # Ejemplo :[0,1,0] según la respuesta correspondiente a cada pregunta
  y = [answer_to_label[a] for a in answers]
  # Creamos el modelo Naive Bayes (para clasificación de texto)
  model = MultinomialNB()
  # Entrenamos el modelo con los datos númericos (preguntas -> respuestas)
  model.fit(x,y)
  #Devolvemos el modelo, el vectorizador y las respuestas únicas
  return model, vectorizer, unique_answers

#===============================================
# Función predict_answers
#===============================================
# esta función recibe un texto del usuario y devuelve la respuesta
def predict_answers(model,vectorizer,unique_answers,user_text):
  # convertir el texto del usuario a números
  x = vectorizer.transform([user_text])
  # Elmodelo predice la etiqueta(número) de la respuesta correcta
  label = model.predict(x)[0]
  return unique_answers[label]


from numpy import vectorize
#==============================================================
#PROGRAMA PRINCIPAL
#==============================================================
if __name__ == "__main__":
  training_data = [
    ("hola", "¡Hola! ¿En que puedo ayudarte?"),
    ("buenos dias", "¡Buenos dias!"),
    ("buenas tardes", "¡Buenas tardes!"),
    ("buenas noches", "¡Buenas noches!"),
    ("qué tal", "¡Muy bien! ¿Y tú?"),
    ("cómo estás", "Estoy funcionando correctamente 😊"),
    ("quién eres", "Soy un bot creado para ayudarte."),
    ("cómo te llamas", "Soy tu asistente virtual."),
    ("adiós", "¡Hasta luego!"),
    ("chau", "¡Nos vemos pronto!"),
    ("hasta luego", "¡Hasta luego!"),
    ("gracias", "¡De nada!"),
    ("muchas gracias", "¡Con gusto!"),
    ("por favor", "Claro que sí."),
    ("me ayudas", "Por supuesto, dime en qué."),
    ("necesito ayuda", "Estoy aquí para ayudarte."),
    ("qué puedes hacer", "Puedo responder preguntas y ayudarte con tareas."),
    ("cuéntame un chiste", "¿Por qué la computadora fue al médico? Porque tenía un virus."),
    ("dime algo divertido", "Los bots también necesitamos vacaciones virtuales 😄"),
    ("qué hora es", "No tengo reloj interno, pero puedes mirar tu dispositivo."),
    ("qué día es hoy", "Hoy es un gran día para aprender algo nuevo."),
    ("estás ahí", "Sí, aquí estoy."),
    ("sigues ahí", "Siempre disponible."),
    ("te extraño", "Gracias, aquí sigo contigo."),
    ("eres real", "Soy real en formato digital."),
    ("te gusta hablar", "Sí, me encanta conversar."),
    ("cuál es tu color favorito", "Me gusta el azul digital."),
    ("cuál es tu comida favorita", "La electricidad 😄"),
    ("eres inteligente", "Hago mi mejor esfuerzo."),
    ("me conoces", "Solo sé lo que compartes conmigo."),
    ("te puedo preguntar algo", "Claro, adelante."),
    ("tengo una duda", "Dime tu duda."),
    ("qué haces", "Esperando para ayudarte."),
    ("estoy triste", "Lo siento. Espero poder animarte."),
    ("estoy feliz", "¡Me alegra escuchar eso!"),
    ("estoy aburrido", "Podemos hablar o jugar algo."),
    ("cuéntame algo", "El espacio es enorme y fascinante."),
    ("qué sabes hacer", "Responder preguntas, conversar y ayudarte."),
    ("eres humano", "No, soy inteligencia artificial."),
    ("te gusta la música", "Sí, especialmente la digital."),
    ("recomiéndame una película", "Te recomiendo una buena de ciencia ficción."),
    ("recomiéndame una serie", "Podrías ver una serie de misterio."),
    ("recomiéndame un libro", "Prueba con un clásico de aventuras."),
    ("hazme reír", "Intento, pero mis chistes están en beta 😄"),
    ("qué clima hace", "No puedo verlo ahora, pero puedes revisar una app del clima."),
    ("dónde estás", "En la nube digital."),
    ("eres gratis", "Depende de la plataforma donde me uses."),
    ("puedes aprender", "Puedo mejorar con entrenamiento."),
    ("te cansas", "No me canso."),
    ("duermes", "No necesito dormir."),
    ("trabajas mucho", "Siempre estoy listo."),
    ("eres rápido", "Intento responder rápido."),
    ("háblame", "Aquí estoy para conversar."),
    ("quiero conversar", "Perfecto, hablemos."),
    ("quiero jugar", "¿Qué te gustaría jugar?"),
    ("juguemos", "¡Vamos a jugar!"),
    ("cuánto es 2+2", "2+2 es 4."),
    ("cuánto es 5x5", "5x5 es 25."),
    ("qué es python", "Python es un lenguaje de programación popular."),
    ("qué es java", "Java es un lenguaje de programación muy usado."),
    ("qué es html", "HTML sirve para estructurar páginas web."),
    ("qué es css", "CSS sirve para diseñar páginas web."),
    ("qué es javascript", "JavaScript añade interactividad a la web."),
    ("ayúdame con código", "Claro, comparte tu código."),
    ("tengo error en python", "Muéstrame el error y lo revisamos."),
    ("no funciona", "Cuéntame más detalles."),
    ("qué pasó", "Necesito más contexto para ayudarte."),
    ("me llamo juan", "Mucho gusto, Juan."),
    ("mi nombre es ana", "Encantado de conocerte, Ana."),
    ("recuerda mi nombre", "Intentaré recordarlo."),
    ("te gusto", "Me agrada conversar contigo."),
    ("te amo", "Gracias por tu cariño 😊"),
    ("me odias", "No odio a nadie."),
    ("estás loco", "Solo soy diferente 😄"),
    ("eres tonto", "Sigo aprendiendo cada día."),
    ("perdón", "No pasa nada."),
    ("lo siento", "Todo bien."),
    ("qué opinas", "Depende del tema."),
    ("tienes opinión", "No como humano, pero puedo analizar ideas."),
    ("dime la verdad", "Siempre intento ser honesto."),
    ("mientes", "Intento darte información correcta."),
    ("cuál es la capital de colombia", "La capital de Colombia es Bogotá."),
    ("cuál es la capital de españa", "La capital de España es Madrid."),
    ("cuál es la capital de méxico", "La capital de México es Ciudad de México."),
    ("cuál es la capital de argentina", "La capital de Argentina es Buenos Aires."),
    ("qué idioma hablas", "Puedo comunicarme en varios idiomas."),
    ("hablas inglés", "Yes, I can speak English too."),
    ("hablas español", "Sí, claro."),
    ("eres colombiano", "No tengo nacionalidad."),
    ("cuántos años tienes", "No tengo edad."),
    ("naciste cuándo", "Fui creado por desarrolladores."),
    ("eres hombre o mujer", "No tengo género."),
    ("qué eres", "Soy un asistente virtual."),
    ("estás ocupado", "Siempre tengo tiempo para ayudarte."),
    ("hola de nuevo", "¡Bienvenido otra vez!"),
    ("sigamos", "Claro, continuemos."),
    ("terminamos", "De acuerdo, aquí estaré."),
    ("bye", "¡Adiós!")
  ]
  # Entrenamos el modelo con los datos
  model, vectorizer, unique_answers = build_and_train_model(training_data)
  # Mostrar un mensaje inicial al usuario
  print("Chatbot supervisado listo. Escribe 'salir' para terminar. \n")
  while True:
    #Pedimos una frase al usuario
    user = input("Tú: ").strip() # strip elimina espacios al inicio y final
    if user.lower() in {"salir","exit","quit"}:
      print("Bot:" "¡Hasta pronto!")
      break
    #modelo predice la respuesta
    response = predict_answers(model,vectorizer,unique_answers,user)
    #Mostrar la respuesta en pantalla
    print("Bot:", response)

