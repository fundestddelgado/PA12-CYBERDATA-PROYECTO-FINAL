import os
from google import genai
from dotenv import load_dotenv

try:
 load_dotenv("Hackaton SIC 2025/API_KEY.env")
except Exception as FILE_NOT_FOUND:
        print(f"ERROR, LLAVE DE API NO ENCONTRADA: {FILE_NOT_FOUND}")

print("Llave cargada")
client= genai.Client(api_key=os.getenv("key"))

#System instructions
prompt_Maestro = """
Tu eres un asistente experto en temas de automóviles, mecanica y leyes de transito 
principalmente de Panamá. Responde de manera clara, concisa y profesional a las 
preguntas de los usuarios, si citas alguna ley o reglamento cita alguna fuente 
oficial de Panamá.
"""

def obtener_respuesta(pregunta):
 response = client.models.generate_content(
    model="gemini-2.5-flash",
    config={
        "system_instruction": prompt_Maestro,
        "temperature": 0.3 # Baja temperatura para que sea más preciso con las leyes
    },
    contents=pregunta
)
 return response.text







