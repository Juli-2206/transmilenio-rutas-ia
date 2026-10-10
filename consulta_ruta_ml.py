"""
=================================================================
SISTEMA INTELIGENTE DE RUTAS - TRANSMILENIO BOGOTÁ
Módulo Combinado: A* + Árbol de Decisión
=================================================================
Universidad : Corporación Universitaria Iberoamericana
Materia     : Inteligencia Artificial
Referencias :
  - Benítez, R. (2014). IA Avanzada. UOC. (Cap. 9 - A*)
  - Palma Méndez, J.T. (2008). Inteligencia artificial:
    métodos, técnicas y aplicaciones. McGraw-Hill. (Cap. 17)

Descripción:
  Encuentra la ruta óptima entre dos estaciones con A*
  y luego predice la categoría de tiempo con el árbol
  de decisión entrenado sobre el dataset de TransMilenio.
=================================================================
"""

import heapq
import math
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════════
# BASE DE CONOCIMIENTO (A*)
# ══════════════════════════════════════════════════════════════════

ESTACIONES = {
    "Portal Norte":              (4.759, -74.047),
    "Toberin":                   (4.740, -74.047),
    "Cardio Infantil":           (4.728, -74.047),
    "Mazuren":                   (4.718, -74.047),
    "Calle 100":                 (4.697, -74.047),
    "Calle 72":                  (4.678, -74.048),
    "Calle 45":                  (4.658, -74.066),
    "Marly":                     (4.645, -74.066),
    "Calle 26":                  (4.626, -74.066),
    "Av Jimenez":                (4.613, -74.072),
    "Las Aguas":                 (4.610, -74.065),
    "Bicentenario":              (4.601, -74.066),
    "Santa Isabel":              (4.587, -74.066),
    "NQS Sur":                   (4.572, -74.066),
    "Portal Sur":                (4.557, -74.140),
    "Portal Americas":           (4.628, -74.188),
    "Patio Bonito":              (4.631, -74.170),
    "Tintal":                    (4.633, -74.160),
    "Banderas":                  (4.631, -74.146),
    "Marsella":                  (4.628, -74.137),
    "General Santander":         (4.628, -74.128),
    "Americas Carrera 30":       (4.628, -74.116),
    "Ricaurte":                  (4.609, -74.091),
    "Portal 80":                 (4.698, -74.133),
    "Quirigua":                  (4.698, -74.120),
    "Bolivia":                   (4.698, -74.108),
    "Granja Carrera 77":         (4.697, -74.098),
    "Minuto de Dios":            (4.697, -74.088),
    "Av Rojas":                  (4.697, -74.078),
    "El Tiempo":                 (4.697, -74.065),
    "Escuela Militar":           (4.680, -74.053),
    "Portal Suba":               (4.742, -74.093),
    "Suba Calle 100":            (4.715, -74.084),
    "Niza Calle 127":            (4.703, -74.072),
    "Pepe Sierra":               (4.683, -74.063),
}

TRONCAL_ESTACION = {
    "Portal Norte": "Caracas",      "Toberin": "Caracas",
    "Cardio Infantil": "Caracas",   "Mazuren": "Caracas",
    "Calle 100": "Caracas",         "Calle 72": "Caracas",
    "Calle 45": "Caracas",          "Marly": "Caracas",
    "Calle 26": "Caracas",          "Av Jimenez": "Caracas",
    "Las Aguas": "Caracas",         "Bicentenario": "Caracas",
    "Santa Isabel": "Caracas",      "NQS Sur": "Caracas",
    "Portal Sur": "Caracas",
    "Portal Americas": "Americas",  "Patio Bonito": "Americas",
    "Tintal": "Americas",           "Banderas": "Americas",
    "Marsella": "Americas",         "General Santander": "Americas",
    "Americas Carrera 30": "Americas", "Ricaurte": "Americas",
    "Portal 80": "Calle80",         "Quirigua": "Calle80",
    "Bolivia": "Calle80",           "Granja Carrera 77": "Calle80",
    "Minuto de Dios": "Calle80",    "Av Rojas": "Calle80",
    "El Tiempo": "Calle80",         "Escuela Militar": "Calle80",
    "Portal Suba": "Suba",          "Suba Calle 100": "Suba",
    "Niza Calle 127": "Suba",       "Pepe Sierra": "Suba",
}

REGLAS_CONEXION = [
    ("Portal Norte",     "Toberin",            4,  "Caracas"),
    ("Toberin",          "Cardio Infantil",     3,  "Caracas"),
    ("Cardio Infantil",  "Mazuren",             3,  "Caracas"),
    ("Mazuren",          "Calle 100",           3,  "Caracas"),
    ("Calle 100",        "Calle 72",            4,  "Caracas"),
    ("Calle 72",         "Calle 45",            5,  "Caracas"),
    ("Calle 45",         "Marly",               3,  "Caracas"),
    ("Marly",            "Calle 26",            4,  "Caracas"),
    ("Calle 26",         "Av Jimenez",          3,  "Caracas"),
    ("Av Jimenez",       "Las Aguas",           2,  "Caracas"),
    ("Las Aguas",        "Bicentenario",        3,  "Caracas"),
    ("Bicentenario",     "Santa Isabel",        4,  "Caracas"),
    ("Santa Isabel",     "NQS Sur",             4,  "Caracas"),
    ("NQS Sur",          "Portal Sur",          5,  "Caracas"),
    ("Portal Americas",      "Patio Bonito",           4,  "Americas"),
    ("Patio Bonito",         "Tintal",                  3,  "Americas"),
    ("Tintal",               "Banderas",                3,  "Americas"),
    ("Banderas",             "Marsella",                3,  "Americas"),
    ("Marsella",             "General Santander",       3,  "Americas"),
    ("General Santander",    "Americas Carrera 30",     3,  "Americas"),
    ("Americas Carrera 30",  "Av Jimenez",              8,  "Americas"),
    ("Av Jimenez",           "Ricaurte",                4,  "Americas"),
    ("Portal 80",        "Quirigua",            3,  "Calle80"),
    ("Quirigua",         "Bolivia",             3,  "Calle80"),
    ("Bolivia",          "Granja Carrera 77",   3,  "Calle80"),
    ("Granja Carrera 77","Minuto de Dios",      3,  "Calle80"),
    ("Minuto de Dios",   "Av Rojas",            3,  "Calle80"),
    ("Av Rojas",         "El Tiempo",           3,  "Calle80"),
    ("El Tiempo",        "Calle 72",            5,  "Calle80"),
    ("Escuela Militar",  "Calle 100",           5,  "Calle80"),
    ("Escuela Militar",  "El Tiempo",           4,  "Calle80"),
    ("Portal Suba",      "Suba Calle 100",      5,  "Suba"),
    ("Suba Calle 100",   "Niza Calle 127",      4,  "Suba"),
    ("Niza Calle 127",   "Pepe Sierra",         4,  "Suba"),
    ("Pepe Sierra",      "Calle 100",           4,  "Suba"),
]

TIEMPO_TRANSBORDO = 5
NOMBRES_CLASES    = ["Rápida", "Media", "Larga"]
EMOJI_CLASE       = ["🟢", "🟡", "🔴"]

HORAS = {
    0: "Madrugada (00:00–05:59)",
    1: "Mañana pico (06:00–09:59)",
    2: "Mañana (10:00–11:59)",
    3: "Tarde (12:00–15:59)",
    4: "Tarde pico (16:00–19:59)",
    5: "Noche (20:00–23:59)",
}

DIAS = {
    0: "Día hábil (lunes–viernes)",
    1: "Sábado",
    2: "Domingo / festivo",
}

# ══════════════════════════════════════════════════════════════════
# MÓDULO 1: GRAFO Y ALGORITMO A*
# ══════════════════════════════════════════════════════════════════

def construir_grafo(reglas):
    grafo = {}
    for origen, destino, tiempo, troncal in reglas:
        for nodo in [origen, destino]:
            if nodo not in grafo:
                grafo[nodo] = []
        grafo[origen].append((destino, tiempo, troncal))
        grafo[destino].append((origen, tiempo, troncal))
    return grafo


def heuristica(est_actual, est_destino):
    lat1, lon1 = ESTACIONES[est_actual]
    lat2, lon2 = ESTACIONES[est_destino]
    dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return dist * 111 * 2


def busqueda_a_estrella(grafo, origen, destino):
    if origen not in grafo:
        return None, f"❌ Estación '{origen}' no encontrada."
    if destino not in grafo:
        return None, f"❌ Estación '{destino}' no encontrada."
    if origen == destino:
        return [], "ℹ️  Origen y destino son la misma estación."

    h0    = heuristica(origen, destino)
    nodo0 = (h0, 0, origen, None, [(origen, None, 0)])
    cola  = [nodo0]
    visitados = {}

    while cola:
        f, g, estacion, troncal_actual, ruta = heapq.heappop(cola)

        if estacion in visitados and visitados[estacion] <= g:
            continue
        visitados[estacion] = g

        if estacion == destino:
            return ruta, None

        for vecino, t_viaje, troncal_vecino in grafo.get(estacion, []):
            costo_transbordo = TIEMPO_TRANSBORDO if (troncal_actual and troncal_actual != troncal_vecino) else 0
            g_nuevo = g + t_viaje + costo_transbordo
            f_nuevo = g_nuevo + heuristica(vecino, destino)
            if vecino not in visitados or visitados[vecino] > g_nuevo:
                nueva_ruta = ruta + [(vecino, troncal_vecino, g_nuevo)]
                heapq.heappush(cola, (f_nuevo, g_nuevo, vecino, troncal_vecino, nueva_ruta))

    return None, "❌ No se encontró ruta entre esas estaciones."


def analizar_ruta(ruta):
    """Extrae métricas de la ruta para el árbol de decisión."""
    transbordos = 0
    troncal_anterior = None
    for estacion, troncal, _ in ruta:
        if troncal and troncal_anterior and troncal != troncal_anterior:
            transbordos += 1
        if troncal:
            troncal_anterior = troncal
    num_estaciones = len(ruta)
    tiempo_total   = ruta[-1][2]
    return num_estaciones, transbordos, tiempo_total


# ══════════════════════════════════════════════════════════════════
# MÓDULO 2: ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════

def cargar_y_entrenar(ruta_csv):
    """Carga el dataset y entrena el árbol. Devuelve modelo + encoder."""
    df = pd.read_csv(ruta_csv)

    le = LabelEncoder()
    df['troncal_origen_cod']  = le.fit_transform(df['troncal_origen'])
    df['troncal_destino_cod'] = le.transform(df['troncal_destino'])

    FEATURES = ['troncal_origen_cod', 'troncal_destino_cod',
                'hora_periodo', 'dia_tipo',
                'num_estaciones', 'num_transbordos']

    X = df[FEATURES]
    y = df['categoria_ruta']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    modelo = DecisionTreeClassifier(
        criterion='gini', max_depth=5,
        min_samples_split=10, min_samples_leaf=5,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    acc = modelo.score(X_test, y_test)

    return modelo, le, acc


def predecir_categoria(modelo, le, troncal_origen, troncal_destino,
                       hora_periodo, dia_tipo, num_estaciones, num_transbordos):
    """Predice la categoría de la ruta con el árbol entrenado."""
    try:
        t_orig_cod = le.transform([troncal_origen])[0]
        t_dest_cod = le.transform([troncal_destino])[0]
    except ValueError:
        # Si una troncal no estaba en el encoder, usar la más cercana
        t_orig_cod = 0
        t_dest_cod = 0

    X_pred = pd.DataFrame([[t_orig_cod, t_dest_cod,
                             hora_periodo, dia_tipo,
                             num_estaciones, num_transbordos]],
                           columns=['troncal_origen_cod', 'troncal_destino_cod',
                                    'hora_periodo', 'dia_tipo',
                                    'num_estaciones', 'num_transbordos'])
    pred  = modelo.predict(X_pred)[0]
    proba = modelo.predict_proba(X_pred)[0]
    return pred, proba


# ══════════════════════════════════════════════════════════════════
# MÓDULO 3: INTERFAZ DE CONSULTA
# ══════════════════════════════════════════════════════════════════

def mostrar_estaciones():
    print("\n" + "─" * 58)
    print("  ESTACIONES DISPONIBLES")
    print("─" * 58)
    por_troncal = {"Caracas": [], "Americas": [], "Calle80": [], "Suba": []}
    for est, troncal in TRONCAL_ESTACION.items():
        por_troncal[troncal].append(est)

    for troncal, ests in por_troncal.items():
        print(f"\n  📍 Troncal {troncal}:")
        for est in ests:
            print(f"       • {est}")
    print("─" * 58)


def pedir_hora():
    print("\n  ⏰ Período horario:")
    for k, v in HORAS.items():
        print(f"     {k} → {v}")
    while True:
        try:
            h = int(input("  Ingresa el número (0-5): ").strip())
            if 0 <= h <= 5:
                return h
            print("  ⚠️  Ingresa un número entre 0 y 5.")
        except ValueError:
            print("  ⚠️  Solo números enteros.")


def pedir_dia():
    print("\n  📅 Tipo de día:")
    for k, v in DIAS.items():
        print(f"     {k} → {v}")
    while True:
        try:
            d = int(input("  Ingresa el número (0-2): ").strip())
            if 0 <= d <= 2:
                return d
            print("  ⚠️  Ingresa un número entre 0 y 2.")
        except ValueError:
            print("  ⚠️  Solo números enteros.")


def mostrar_resultado(ruta, origen, destino, hora, dia, pred, proba,
                      num_estaciones, num_transbordos, tiempo_total):
    """Muestra la ruta de A* y la predicción del árbol integradas."""

    print("\n" + "═" * 62)
    print("   🚌  RUTA ÓPTIMA  —  A* + Árbol de Decisión")
    print("═" * 62)
    print(f"  De    : {origen}")
    print(f"  A     : {destino}")
    print(f"  Hora  : {HORAS[hora]}")
    print(f"  Día   : {DIAS[dia]}")
    print("─" * 62)

    # Tramo por tramo
    troncal_anterior = None
    for i, (estacion, troncal, t_acum) in enumerate(ruta):
        if i == 0:
            print(f"\n  🟢 SALIDA  → {estacion}  [{TRONCAL_ESTACION.get(estacion,'')}]")
            troncal_anterior = troncal
            continue
        if troncal and troncal_anterior and troncal != troncal_anterior:
            print(f"\n  🔄 TRANSBORDO en {estacion}")
            print(f"     Aborda Troncal {troncal}  (+{TIEMPO_TRANSBORDO} min)")
            troncal_anterior = troncal
        elif i == len(ruta) - 1:
            print(f"\n  🔴 LLEGADA → {estacion}  ✓")
        else:
            print(f"     ➡  {estacion}  [{troncal}]")

    print("\n" + "─" * 62)
    print(f"  ⏱  Tiempo estimado A*    : {tiempo_total} minutos")
    print(f"  🚉 Estaciones recorridas : {num_estaciones}")
    print(f"  🔄 Transbordos           : {num_transbordos}")

    # Predicción del árbol
    clase     = NOMBRES_CLASES[pred]
    emoji     = EMOJI_CLASE[pred]
    rangos    = ["< 25 min", "25 – 50 min", "> 50 min"]
    print("\n" + "─" * 62)
    print("  🤖  PREDICCIÓN — Árbol de Decisión (CART / Gini)")
    print("─" * 62)
    print(f"  Categoría predicha : {emoji}  {clase}  ({rangos[pred]})")
    print(f"\n  Confianza por clase:")
    for i, (nombre, p) in enumerate(zip(NOMBRES_CLASES, proba)):
        barra = "█" * int(p * 30)
        marca = " ◀ predicción" if i == pred else ""
        print(f"    {EMOJI_CLASE[i]}  {nombre:7s}  {p:5.1%}  {barra}{marca}")
    print("═" * 62)


# ══════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main():
    # Localizar el dataset relativo a este script
    dir_script = os.path.dirname(os.path.abspath(__file__))
    ruta_csv   = os.path.join(dir_script, "dataset_transmilenio.csv")

    print("\n" + "=" * 62)
    print("   SISTEMA INTELIGENTE DE RUTAS — TRANSMILENIO BOGOTÁ")
    print("   A* (Benítez 2014, Cap. 9) + Árbol de Decisión")
    print("   (Palma Méndez 2008, Cap. 17)")
    print("=" * 62)

    # Cargar y entrenar el árbol
    print("\n  ⚙️  Cargando dataset y entrenando árbol de decisión...")
    try:
        modelo, le, acc = cargar_y_entrenar(ruta_csv)
        print(f"  ✅ Árbol entrenado — Exactitud en prueba: {acc*100:.2f}%")
    except FileNotFoundError:
        print(f"\n  ❌ No se encontró '{ruta_csv}'.")
        print("     Ejecuta primero: python generar_dataset.py")
        return

    grafo = construir_grafo(REGLAS_CONEXION)

    while True:
        mostrar_estaciones()

        print("\n  (Escribe 'salir' para terminar)\n")
        origen = input("  📍 Estación de ORIGEN  : ").strip()
        if origen.lower() == "salir":
            break

        destino = input("  📍 Estación de DESTINO : ").strip()
        if destino.lower() == "salir":
            break

        hora = pedir_hora()
        dia  = pedir_dia()

        print("\n  🔍 Calculando ruta óptima con A*...")
        ruta, error = busqueda_a_estrella(grafo, origen, destino)

        if error:
            print(f"\n  {error}")
        else:
            num_est, num_trans, tiempo = analizar_ruta(ruta)

            troncal_orig = TRONCAL_ESTACION.get(origen, "Caracas")
            troncal_dest = TRONCAL_ESTACION.get(destino, "Caracas")

            pred, proba = predecir_categoria(
                modelo, le,
                troncal_orig, troncal_dest,
                hora, dia,
                num_est, num_trans
            )

            mostrar_resultado(
                ruta, origen, destino, hora, dia,
                pred, proba, num_est, num_trans, tiempo
            )

        print("\n  ¿Consultar otra ruta? (Enter = sí / 'salir' = no)")
        if input("  → ").strip().lower() == "salir":
            break

    print("\n  ¡Hasta luego!\n")


if __name__ == "__main__":
    main()
