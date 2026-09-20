# Sistema Inteligente de Rutas - TransMilenio Bogotá

**Materia:** Inteligencia Artificial  
**Universidad:** Corporación Universitaria Iberoamericana  
**Referencia teórica:** Benítez, R. (2014). *Inteligencia artificial avanzada*. Barcelona: Editorial UOC.
- Capítulo 2: Lógica y representación del conocimiento  
- Capítulo 3: Sistemas basados en reglas  
- Capítulo 9: Técnicas basadas en búsquedas heurísticas

---

## Descripción del proyecto

Sistema inteligente que encuentra la **ruta óptima** (menor tiempo) entre dos estaciones de TransMilenio en Bogotá, implementado en Python usando:

- **Base de conocimiento** con hechos (estaciones) y reglas lógicas (conexiones)
- **Motor de inferencia** basado en reglas IF-THEN
- **Algoritmo A\*** (búsqueda heurística) para encontrar la ruta de menor costo

---

## Requisitos

- Python 3.7 o superior  
- No requiere librerías externas (solo `heapq` y `math` de la librería estándar)

---

## Instalación y ejecución

### 1. Verifica que Python esté instalado

```bash
python --version
```

Debe mostrar: `Python 3.x.x`

### 2. Descarga o clona el repositorio

```bash
git clone https://github.com/Juli-2206/transmilenio-rutas-ia.git
cd transmilenio-rutas
```

### 3. Ejecuta el programa

```bash
python transmilenio_rutas.py
```

En Windows también puedes usar:
```bash
py transmilenio_rutas.py
```

---

## Uso del sistema

1. Al iniciar, el programa muestra todas las estaciones disponibles.
2. Ingresa el nombre exacto de la estación de **origen**.
3. Ingresa el nombre exacto de la estación de **destino**.
4. El sistema calcula y muestra la ruta óptima con:
   - Secuencia de estaciones
   - Troncal a abordar en cada tramo
   - Puntos de transbordo
   - Tiempo total estimado en minutos

---

## Ejemplos de rutas para probar

| Origen | Destino | Transbordos |
|--------|---------|-------------|
| Portal Norte | Portal Sur | 0 |
| Portal Americas | Portal Norte | 1 |
| Portal 80 | Ricaurte | 2 |
| Portal Suba | Portal Sur | 1 |

---

## Estructura del código

```
transmilenio_rutas.py
│
├── MÓDULO 1 – BASE DE CONOCIMIENTO
│   ├── ESTACIONES    → Hechos: nodos del grafo con coordenadas
│   └── REGLAS_CONEXION → Reglas: aristas con tiempo y troncal
│
├── MÓDULO 2 – SISTEMA BASADO EN REGLAS
│   ├── construir_grafo()         → Aplica regla de simetría
│   ├── regla_costo_transbordo()  → IF cambio_troncal THEN +5 min
│   ├── regla_estacion_valida()   → IF estacion ∈ KB THEN válida
│   └── regla_mismo_origen_destino() → IF origen=destino THEN trivial
│
├── MÓDULO 3 – BÚSQUEDA HEURÍSTICA A*
│   ├── heuristica()              → h(n): distancia euclidiana
│   └── busqueda_a_estrella()     → f(n) = g(n) + h(n)
│
└── MÓDULO 4 – PRESENTACIÓN
    ├── mostrar_estaciones()
    └── mostrar_ruta()
```

---

## Arquitectura del sistema

```
Entrada (origen, destino)
        ↓
[VALIDACIÓN POR REGLAS]
  Regla 2: ¿estación existe?
  Regla 3: ¿origen ≠ destino?
        ↓
[BÚSQUEDA A*]
  Para cada nodo expandido:
    g(n) = tiempo acumulado real
    h(n) = heurística euclidiana
    Regla 1: ¿transbordo? → +5 min
        ↓
Salida: Ruta óptima + tiempo total
```

---

## Fundamento teórico

### Representación del conocimiento (Cap. 2)
Las estaciones son **hechos** y las conexiones son **reglas** del tipo:
```
conexion(A, B, tiempo, troncal) → puede_viajar(A, B)
```

### Sistema basado en reglas (Cap. 3)
El motor de inferencia aplica reglas IF-THEN:
```
SI troncal_actual ≠ troncal_siguiente ENTONCES tiempo += 5
SI estacion ∉ base_conocimiento       ENTONCES error
```

### Búsqueda heurística A* (Cap. 9)
Función de evaluación: `f(n) = g(n) + h(n)`
- `g(n)`: costo real (minutos acumulados de viaje)
- `h(n)`: heurística admisible (distancia euclidiana estimada)
- La admisibilidad garantiza que A* encuentra siempre la solución óptima

