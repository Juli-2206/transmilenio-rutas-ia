"""
=================================================================
SISTEMA INTELIGENTE DE RUTAS - TRANSMILENIO BOGOTÁ
=================================================================
Autor  : Juliana Esther Naranjo Camacho
Materia  : Inteligencia Artificial
Universidad: Corporación Universitaria Iberoamericana

Fundamentos teóricos (Benítez, R. - IA Avanzada, UOC 2014):
  - Capítulo 2 : Lógica y representación del conocimiento
  - Capítulo 3 : Sistemas basados en reglas
  - Capítulo 9 : Técnicas basadas en búsquedas heurísticas

Descripción:
  Sistema que encuentra la mejor ruta entre dos estaciones de
  TransMilenio usando una base de conocimiento en reglas lógicas
  y el algoritmo de búsqueda heurística A* (A-estrella).
=================================================================
"""

import heapq
import math


# =================================================================
# MÓDULO 1: BASE DE CONOCIMIENTO
# Ref: Capítulo 2 - Lógica y representación del conocimiento
#
# HECHOS    → Estaciones con sus coordenadas geográficas
# REGLAS    → Conexiones entre estaciones (troncal + tiempo)
# =================================================================

# HECHOS: Estaciones del sistema (nombre → coordenadas)
ESTACIONES = {
    # --- Troncal Caracas / Norte (eje norte-sur) ---
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

    # --- Troncal Américas (eje occidente) ---
    "Portal Americas":           (4.628, -74.188),
    "Patio Bonito":              (4.631, -74.170),
    "Tintal":                    (4.633, -74.160),
    "Banderas":                  (4.631, -74.146),
    "Marsella":                  (4.628, -74.137),
    "General Santander":         (4.628, -74.128),
    "Americas Carrera 30":       (4.628, -74.116),
    "Ricaurte":                  (4.609, -74.091),

    # --- Troncal Calle 80 (eje noroccidente) ---
    "Portal 80":                 (4.698, -74.133),
    "Quirigua":                  (4.698, -74.120),
    "Bolivia":                   (4.698, -74.108),
    "Granja Carrera 77":         (4.697, -74.098),
    "Minuto de Dios":            (4.697, -74.088),
    "Av Rojas":                  (4.697, -74.078),
    "El Tiempo":                 (4.697, -74.065),
    "Escuela Militar":           (4.680, -74.053),

    # --- Troncal Suba (eje noroccidente) ---
    "Portal Suba":               (4.742, -74.093),
    "Suba Calle 100":            (4.715, -74.084),
    "Niza Calle 127":            (4.703, -74.072),
    "Pepe Sierra":               (4.683, -74.063),
}


# REGLAS DE CONEXIÓN:
# Formato → (estacion_A, estacion_B, tiempo_minutos, troncal)
# Semántica lógica:
#   SI existe_conexion(A, B, T, tiempo) ENTONCES puede_viajar(A, B, tiempo)
#   Y puede_viajar(B, A, tiempo)   ← regla de simetría

REGLAS_CONEXION = [
    # Troncal Caracas Norte/Sur
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

    # Troncal Américas
    ("Portal Americas",      "Patio Bonito",           4,  "Americas"),
    ("Patio Bonito",         "Tintal",                  3,  "Americas"),
    ("Tintal",               "Banderas",                3,  "Americas"),
    ("Banderas",             "Marsella",                3,  "Americas"),
    ("Marsella",             "General Santander",       3,  "Americas"),
    ("General Santander",    "Americas Carrera 30",     3,  "Americas"),
    ("Americas Carrera 30",  "Av Jimenez",              8,  "Americas"),
    ("Av Jimenez",           "Ricaurte",                4,  "Americas"),

    # Troncal Calle 80
    ("Portal 80",        "Quirigua",            3,  "Calle80"),
    ("Quirigua",         "Bolivia",             3,  "Calle80"),
    ("Bolivia",          "Granja Carrera 77",   3,  "Calle80"),
    ("Granja Carrera 77","Minuto de Dios",      3,  "Calle80"),
    ("Minuto de Dios",   "Av Rojas",            3,  "Calle80"),
    ("Av Rojas",         "El Tiempo",           3,  "Calle80"),
    ("El Tiempo",        "Calle 72",            5,  "Calle80"),
    ("Escuela Militar",  "Calle 100",           5,  "Calle80"),
    ("Escuela Militar",  "El Tiempo",           4,  "Calle80"),

    # Troncal Suba
    ("Portal Suba",      "Suba Calle 100",      5,  "Suba"),
    ("Suba Calle 100",   "Niza Calle 127",      4,  "Suba"),
    ("Niza Calle 127",   "Pepe Sierra",         4,  "Suba"),
    ("Pepe Sierra",      "Calle 100",           4,  "Suba"),
]


# =================================================================
# MÓDULO 2: SISTEMA BASADO EN REGLAS
# Ref: Capítulo 3 - Sistemas basados en reglas
#
# Motor de inferencia con reglas IF-THEN que gobiernan
# la lógica de movimiento por la red de transporte
# =================================================================

TIEMPO_TRANSBORDO = 5   # minutos al cambiar de troncal
TIEMPO_ESPERA     = 2   # minutos de espera en estación


def construir_grafo(reglas):
    """
    Construye el grafo de adyacencia aplicando la regla de simetría:
    REGLA: SI conexion(A, B) ENTONCES conexion(B, A)
    """
    grafo = {}
    for origen, destino, tiempo, troncal in reglas:
        for nodo in [origen, destino]:
            if nodo not in grafo:
                grafo[nodo] = []
        grafo[origen].append((destino, tiempo, troncal))
        grafo[destino].append((origen, tiempo, troncal))
    return grafo


def regla_costo_transbordo(troncal_actual, troncal_siguiente):
    """
    REGLA 1 (Transbordo):
      SI troncal_actual ≠ troncal_siguiente
      ENTONCES costo_adicional = TIEMPO_TRANSBORDO
      SINO     costo_adicional = 0
    """
    if troncal_actual is not None and troncal_actual != troncal_siguiente:
        return TIEMPO_TRANSBORDO
    return 0


def regla_estacion_valida(estacion, grafo):
    """
    REGLA 2 (Validación):
      SI estacion ∈ base_de_conocimiento
      ENTONCES estacion_valida = VERDADERO
      SINO     estacion_valida = FALSO
    """
    return estacion in grafo


def regla_mismo_origen_destino(origen, destino):
    """
    REGLA 3 (Trivialidad):
      SI origen = destino
      ENTONCES no_se_requiere_ruta = VERDADERO
    """
    return origen == destino


# =================================================================
# MÓDULO 3: BÚSQUEDA HEURÍSTICA A*
# Ref: Capítulo 9 - Técnicas basadas en búsquedas heurísticas
#
# f(n) = g(n) + h(n)
#   g(n) = costo real acumulado (tiempo en minutos)
#   h(n) = heurística admisible (distancia euclidiana convertida)
# =================================================================

def heuristica(estacion_actual, estacion_destino):
    """
    Función heurística h(n): distancia euclidiana entre coordenadas.

    Propiedad de admisibilidad: nunca sobreestima el costo real,
    ya que la distancia en línea recta siempre es ≤ distancia real.

    Conversión: 1 grado ≈ 111 km; velocidad promedio ≈ 30 km/h
    → factor ≈ 2 min/grado (subestimación conservadora)
    """
    lat1, lon1 = ESTACIONES[estacion_actual]
    lat2, lon2 = ESTACIONES[estacion_destino]
    distancia_grados = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return distancia_grados * 111 * 2   # estimación en minutos


def busqueda_a_estrella(grafo, origen, destino):
    """
    Algoritmo A* para encontrar la ruta de menor tiempo.

    Estado : (f, g, estacion, troncal_actual, ruta)
      f    : costo total estimado = g + h
      g    : tiempo real acumulado en minutos
      ruta : lista de (estacion, troncal, tiempo_acumulado)

    Garantía: si h(n) es admisible, A* encuentra siempre
    la solución óptima (menor tiempo de viaje).
    """
    # --- Validaciones mediante reglas ---
    if not regla_estacion_valida(origen, grafo):
        return None, f"❌ La estación '{origen}' no existe. Revisa el nombre."
    if not regla_estacion_valida(destino, grafo):
        return None, f"❌ La estación '{destino}' no existe. Revisa el nombre."
    if regla_mismo_origen_destino(origen, destino):
        return [], "ℹ️  Origen y destino son la misma estación."

    # --- Inicialización ---
    h0    = heuristica(origen, destino)
    nodo0 = (h0, 0, origen, None, [(origen, None, 0)])
    cola  = [nodo0]          # cola de prioridad (min-heap)
    visitados = {}           # estacion → menor g conocido

    # --- Ciclo principal A* ---
    while cola:
        f, g, estacion, troncal_actual, ruta = heapq.heappop(cola)

        # Ignorar si ya encontramos un camino mejor a esta estación
        if estacion in visitados and visitados[estacion] <= g:
            continue
        visitados[estacion] = g

        # ¿Meta alcanzada?
        if estacion == destino:
            return ruta, None

        # --- Expansión de vecinos ---
        for vecino, tiempo_viaje, troncal_vecino in grafo.get(estacion, []):
            # Aplicar reglas para calcular costo del movimiento
            costo_transbordo = regla_costo_transbordo(troncal_actual, troncal_vecino)
            g_nuevo = g + tiempo_viaje + costo_transbordo
            h_nuevo = heuristica(vecino, destino)
            f_nuevo = g_nuevo + h_nuevo

            # Solo añadir si mejora el costo conocido
            if vecino not in visitados or visitados[vecino] > g_nuevo:
                nueva_ruta = ruta + [(vecino, troncal_vecino, g_nuevo)]
                heapq.heappush(cola, (f_nuevo, g_nuevo, vecino,
                                      troncal_vecino, nueva_ruta))

    return None, "❌ No se encontró ruta entre las estaciones indicadas."


# =================================================================
# MÓDULO 4: PRESENTACIÓN DE RESULTADOS
# =================================================================

def mostrar_estaciones():
    """Lista todas las estaciones disponibles en la base de conocimiento."""
    print("\n" + "─" * 55)
    print("  ESTACIONES DISPONIBLES EN LA BASE DE CONOCIMIENTO")
    print("─" * 55)
    estaciones_ordenadas = sorted(ESTACIONES.keys())
    for i, est in enumerate(estaciones_ordenadas, 1):
        print(f"  {i:2d}. {est}")
    print("─" * 55)


def mostrar_ruta(ruta, origen, destino):
    """Imprime la ruta encontrada con formato legible."""
    print("\n" + "=" * 60)
    print("   🚌  RUTA ÓPTIMA - TRANSMILENIO BOGOTÁ")
    print("=" * 60)
    print(f"  De : {origen}")
    print(f"  A  : {destino}")
    print("─" * 60)

    troncal_anterior = None
    transbordos      = 0

    for i, (estacion, troncal, tiempo_acum) in enumerate(ruta):
        if i == 0:
            print(f"\n  🟢 SALIDA  → {estacion}")
            troncal_anterior = troncal
            continue

        # Detectar transbordo
        if troncal != troncal_anterior and troncal is not None:
            print(f"\n  🔄 TRANSBORDO en {estacion}")
            print(f"     Abordar Troncal {troncal}  (+{TIEMPO_TRANSBORDO} min espera)")
            transbordos += 1
            troncal_anterior = troncal
        elif i == len(ruta) - 1:
            print(f"\n  🔴 LLEGADA → {estacion}  ✓")
        else:
            print(f"     ➡  {estacion}  [Troncal {troncal}]")

    tiempo_total = ruta[-1][2]
    print("\n" + "─" * 60)
    print(f"  ⏱  Tiempo total estimado  : {tiempo_total} minutos")
    print(f"  🔄 Número de transbordos  : {transbordos}")
    print(f"  🚉 Estaciones en la ruta  : {len(ruta)}")
    print("=" * 60)


# =================================================================
# PROGRAMA PRINCIPAL
# =================================================================

def main():
    print("\n" + "=" * 60)
    print("   SISTEMA INTELIGENTE DE RUTAS - TRANSMILENIO BOGOTÁ")
    print("   Algoritmo: Búsqueda heurística A* (A-estrella)")
    print("   Base: Benítez, R. (2014). IA Avanzada. UOC.")
    print("=" * 60)

    # Construir el grafo desde las reglas de conexión
    grafo = construir_grafo(REGLAS_CONEXION)

    while True:
        mostrar_estaciones()

        print("\n  (Escribe 'salir' para terminar)\n")
        origen  = input("  📍 Estación de ORIGEN  : ").strip()
        if origen.lower() == "salir":
            print("\n  ¡Hasta luego!\n")
            break

        destino = input("  📍 Estación de DESTINO : ").strip()
        if destino.lower() == "salir":
            print("\n  ¡Hasta luego!\n")
            break

        print(f"\n  🔍 Calculando ruta óptima con A*...")
        ruta, error = busqueda_a_estrella(grafo, origen, destino)

        if error:
            print(f"\n  {error}")
        else:
            mostrar_ruta(ruta, origen, destino)

        print("\n  ¿Deseas buscar otra ruta? (Enter para continuar / 'salir' para terminar)")
        resp = input("  → ").strip().lower()
        if resp == "salir":
            print("\n  ¡Hasta luego!\n")
            break


if __name__ == "__main__":
    main()
