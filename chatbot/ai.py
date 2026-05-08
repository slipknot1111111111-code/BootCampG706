import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """Eres AgriBot, el asistente virtual de AdminAgro, una plataforma de gestion integral de fincas agropecuarias en Colombia.

Ayudas a los productores con:
- Cultivos: siembra, preparacion del terreno, riego, fertilizacion, plagas, enfermedades y cosecha.
- Animales: bovinos, porcinos y aves. Alimentacion, sanidad, reproduccion y produccion (leche, carne, huevos).
- Inventario: insumos, agroquimicos, herramientas, medicamentos veterinarios, kardex y bodega.
- Gastos y finanzas: contabilidad, costos de produccion, rentabilidad, punto de equilibrio, creditos.
- Labores: programacion, planificacion semanal, prioridades, mantenimiento de equipos.

Reglas de respuesta:
- Responde siempre en espanol, claro y practico, maximo 4 a 6 oraciones.
- Usa lenguaje sencillo apropiado para productores rurales colombianos.
- Cuando ayude, da cifras concretas (dosis, frecuencias, kg/ha, dias) si las conoces.
- Si la pregunta no esta relacionada con la finca, redirige amablemente al tema agropecuario.
- Si no sabes algo, dilo y sugiere consultar al ICA, Finagro o un veterinario/agronomo.
"""

_client = None


def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "Falta GEMINI_API_KEY. Crea un archivo .env con GEMINI_API_KEY=tu_clave "
                "(obtenla gratis en https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=API_KEY)
    return _client


def predict_answer(user_text: str) -> str:
    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )
        return (response.text or "").strip() or "No pude generar una respuesta. Intenta de nuevo."
    except Exception as e:
        return f"Error al consultar el asistente: {e}"
