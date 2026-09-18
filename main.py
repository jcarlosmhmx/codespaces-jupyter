#importación de librerias
import json
import sys
import urllib.request

import pandas as pd
import pysqlite3

sys.modules["sqlite3"] = pysqlite3
from crewai.tools import tool
from crewai import Agent, Task, Crew, LLM


def verificar_ollama():
    """Valida que Ollama esté corriendo y que el modelo requerido exista."""
    url = "http://localhost:11434/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Ollama no está disponible. Debes ejecutar 'ollama serve' y luego 'ollama pull llama3.2'."
        ) from exc

    modelos = {item.get("name", "") for item in data.get("models", [])}
    if "llama3.2:latest" not in modelos and "llama3.2" not in modelos:
        raise RuntimeError(
            "No está instalado el modelo 'llama3.2'. Ejecuta 'ollama pull llama3.2'."
        )


#configuración del modelo del lenguaje
llm = LLM(
    model="ollama/llama3.2",
    api_key="ollama",
    base_url="http://localhost:11434"
)


#Carga los datos de ventas
def cargar_ventas():
    df = pd.read_csv("ventas.csv")
    return df


#Función que devuelva los productos más vendidos
def top_productos(n=5):
    df = cargar_ventas()
    resultado = df.groupby("producto")["ventas"].sum().sort_values(ascending=False)
    return resultado.head(n)


#función que devulve las n-sucurcasles con más ventas
def top_sucursales(n=5):
    df = cargar_ventas()
    resultado = df.groupby("sucursal")["ventas"].sum().sort_values(ascending=False)
    return resultado.head(n)


@tool
def obtener_top_productos():
    """Devuelve los cinco productos con más ventas."""
    return top_productos().to_string()


@tool
def obtener_top_sucursales():
    """Devuelve las cinco sucursales con más ventas."""
    return top_sucursales().to_string()


#Definición de agentes
analista = Agent(
    role="Analista de ventas",
    goal="Analizar los datos de ventas y proporcionar información sobre los productos y sucursales más vendidos.",
    backstory="Analista de ventas con experiencia en análisis de datos y generación de informes.",
    tools=[obtener_top_productos, obtener_top_sucursales],
    verbose=False,
    allow_delegation=False,
    llm=llm,
)

#Tareas
tarea_analisis = Task(
    name="Análisis de ventas",
    description="Usa la herramienta obtener_top_productos para obtener los cinco productos con más ventas y la herramienta obtener_top_sucursales para obtener las cinco sucursales con más ventas.",
    expected_output="Un resumen con los cinco productos más vendidos y las cinco sucursales con más ventas, en formato claro y legible.",
    agent=analista,
    verbose=False,
)

#Configuración del crew
crew = Crew(
    name="Equipo de análisis de ventas",
    description="Equipo encargado de analizar los datos de ventas y proporcionar información sobre los productos y sucursales más vendidos.",
    tasks=[tarea_analisis],
    agents=[analista],
    verbose=False,
)


def main():
    verificar_ollama()
    print("Ejecutando análisis de ventas...")
    resultado = crew.kickoff()
    print("\nResultado:")
    print(resultado)


if __name__ == "__main__":
    main()
