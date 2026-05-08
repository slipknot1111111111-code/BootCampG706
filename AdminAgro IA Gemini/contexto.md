# Contexto del Proyecto — AdminAgro

## Descripción General

Sitio web educativo sobre energías renovables en Colombia, con un chatbot conversacional basado en Machine Learning. Es un MVP funcional orientado a portafolio o uso académico, combinando Flask como backend web con un modelo Naive Bayes entrenado localmente.

---

## Estructura de Directorios

```
C:\AdminAgro\
├── main.py                        # Aplicación Flask principal (orquestador)
├── requirements.txt               # Dependencias Python
├── Procfile                       # Configuración Heroku/Gunicorn
├── .gitignore
│
├── chatbot/
│   ├── __init__.py
│   ├── data.py                    # 102 pares pregunta-respuesta de entrenamiento
│   ├── model.py                   # Lógica ML: entrenamiento, carga y predicción
│   └── static/
│       ├── css/style.css          # Estilos globales
│       ├── js/app.js              # Lógica del chat en frontend
│       └── img/                   # Imágenes (solar, eólica, hidráulica, favicon)
│
├── models/
│   ├── chatbot_model.pkl          # Modelo Naive Bayes serializado
│   ├── vectorizer.pkl             # CountVectorizer serializado
│   └── answers.pkl                # Lista de respuestas únicas
│
└── templates/
    ├── base.html                  # Template base con navbar y widget de chat
    ├── index.html                 # Página de inicio
    ├── energia.html               # Tipos de energía renovable
    ├── beneficios.html            # Beneficios de energías limpias
    ├── contacto.html              # Formulario de contacto
    └── mapa.html                  # Mapa interactivo Leaflet
```

---

## Tecnologías

| Capa | Tecnología |
|------|-----------|
| Backend | Flask 3.1.3, Python 3.12 |
| ML | scikit-learn 1.8.0 (Naive Bayes + CountVectorizer), joblib 1.5.3 |
| Frontend | HTML5, CSS3, JavaScript vanilla, Jinja2 |
| Mapas | Leaflet.js + OpenStreetMap (CDN) |
| Producción | Gunicorn, Heroku (Procfile) |
| Entorno | Python venv (`env312`) |

---

## Rutas de la Aplicación

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Página de inicio: hero section + tarjetas de energías |
| GET | `/energia` | Descripción de 3 tipos de energía renovable |
| GET | `/beneficios` | Lista de beneficios de energías limpias |
| GET | `/contacto` | Formulario de contacto (sin backend implementado) |
| GET | `/mapa` | Mapa interactivo con 3 proyectos energéticos en Colombia |
| POST | `/chat` | API JSON: recibe mensaje, devuelve respuesta del chatbot |

---

## Flujo del Chatbot

```
Usuario escribe en UI (app.js)
  → POST /chat con mensaje (JSON)
    → main.py llama predict_answer()
      → model.py vectoriza con models/vectorizer.pkl
      → model.py predice con models/chatbot_model.pkl
      → Devuelve respuesta de models/answers.pkl
  ← JSON con respuesta
← app.js renderiza burbuja en chat
```

**Datos de entrenamiento**: 102 pares pregunta-respuesta en español (`chatbot/data.py`). Cubre saludos, despedidas, preguntas sobre identidad, chistes y consejos generales.

**Algoritmo**: Multinomial Naive Bayes con CountVectorizer. Los modelos se entrenan una vez y se persisten en `models/`.

---

## Estado del Proyecto

### Completado
- Estructura base de Flask con 5 rutas web
- Chatbot entrenado y funcional (102 interacciones)
- Widget de chat flotante en todas las páginas (via `base.html`)
- Mapa interactivo Leaflet con 3 puntos en Colombia
- Estilos CSS responsivos (breakpoint a 768px)
- Herencia de templates con Jinja2
- Serialización de modelos ML con joblib
- Configuración para deploy en Heroku

### Pendiente / Incompleto
- Formulario de contacto sin backend (no envía emails)
- Sin manejo de errores robusto
- Sin tests unitarios
- Sin sistema de logs
- Sin loading spinner en el chat
- Datos de entrenamiento hardcodeados (sin base de datos)
- Varios templates y archivos estáticos sin commitear (ver git status)

---

## Relaciones Clave entre Archivos

- `main.py` importa de `chatbot.data` y `chatbot.model`
- Todos los templates heredan de `templates/base.html`
- `base.html` carga `chatbot/static/css/style.css` y `chatbot/static/js/app.js`
- `app.js` llama al endpoint `POST /chat` de `main.py`
- `chatbot/model.py` genera y lee los archivos `models/*.pkl`

---

## Git

- **Rama activa**: `main`
- **Último commit**: `3c7b476 v1`
- **Archivos modificados sin commitear**: `main.py`, `templates/index.html`, `.gitignore`
- **Archivos nuevos sin commitear**: `templates/base.html`, `templates/beneficios.html`, `templates/contacto.html`, `templates/energia.html`, `templates/mapa.html`, `chatbot/static/`

---

## Notas de Deploy

- El `Procfile` contiene: `web: gunicorn main:app`
- La aplicación está preparada para Heroku
- El entorno virtual `env312` (Python 3.12) está excluido del repositorio vía `.gitignore`
