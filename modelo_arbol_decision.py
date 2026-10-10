"""
=================================================================
SISTEMA INTELIGENTE DE RUTAS - TRANSMILENIO BOGOTÁ
Módulo de Aprendizaje Automático: Árbol de Decisión
=================================================================
Universidad : Corporación Universitaria Iberoamericana
Materia     : Inteligencia Artificial
Referencia  : Palma Méndez, J. T. (2008). Inteligencia artificial:
              métodos, técnicas y aplicaciones. McGraw-Hill España.
              Capítulo 17: Aprendizaje de árboles y reglas de decisión.

Descripción:
  Modelo de clasificación basado en árbol de decisión (ID3/CART)
  que predice la categoría de una ruta de TransMilenio (Rápida,
  Media o Larga) a partir de características del viaje.

Variables de entrada (features):
  - troncal_origen  : troncal de la estación de origen (0-3)
  - troncal_destino : troncal de la estación de destino (0-3)
  - hora_periodo    : período horario (0=madrugada … 5=noche)
  - dia_tipo        : tipo de día (0=hábil, 1=sábado, 2=festivo)
  - num_estaciones  : número de estaciones en la ruta
  - num_transbordos : número de transbordos necesarios (0-1)

Variable objetivo:
  - categoria_ruta  : 0=Rápida (<25 min), 1=Media (25-50 min),
                      2=Larga (>50 min)
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

print("=" * 65)
print("  MODELO DE ÁRBOL DE DECISIÓN - TRANSMILENIO BOGOTÁ")
print("  Ref: Palma Méndez (2008), Cap. 17")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════
print("\n[1] Cargando dataset...")
df = pd.read_csv("dataset_transmilenio.csv")

print(f"    Registros totales     : {len(df)}")
print(f"    Variables (columnas)  : {df.shape[1]}")
print(f"\n    Distribución de clases:")
dist = df['categoria_nombre'].value_counts()
for cat, n in dist.items():
    print(f"      {cat:10s}: {n:3d} ({n/len(df)*100:.1f}%)")

print(f"\n    Primeras 5 filas:")
print(df[['estacion_origen','troncal_origen','estacion_destino',
          'hora_periodo','num_transbordos','tiempo_minutos',
          'categoria_nombre']].head().to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# 2. PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════
print("\n[2] Preprocesando datos...")

# Codificar troncales como números
le_troncal = LabelEncoder()
df['troncal_origen_cod']  = le_troncal.fit_transform(df['troncal_origen'])
df['troncal_destino_cod'] = le_troncal.transform(df['troncal_destino'])

print(f"    Codificación de troncales: {dict(zip(le_troncal.classes_, le_troncal.transform(le_troncal.classes_)))}")

# Features y target
FEATURES = ['troncal_origen_cod', 'troncal_destino_cod',
            'hora_periodo', 'dia_tipo',
            'num_estaciones', 'num_transbordos']

NOMBRES_FEATURES = ['Troncal Origen', 'Troncal Destino',
                    'Hora Período', 'Día Tipo',
                    'N° Estaciones', 'N° Transbordos']

X = df[FEATURES]
y = df['categoria_ruta']

print(f"    Features seleccionadas: {FEATURES}")
print(f"    Tamaño X: {X.shape}")

# División entrenamiento / prueba (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print(f"    Entrenamiento: {len(X_train)} muestras")
print(f"    Prueba       : {len(X_test)} muestras")

# ══════════════════════════════════════════════════════════════════
# 3. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════
print("\n[3] Entrenando árbol de decisión...")

# Criterio: gini (CART) — Palma Méndez (2008) p. 412
modelo = DecisionTreeClassifier(
    criterion='gini',       # impureza de Gini (CART)
    max_depth=5,            # profundidad máxima para evitar sobreajuste
    min_samples_split=10,   # mínimo de muestras para dividir un nodo
    min_samples_leaf=5,     # mínimo de muestras en una hoja
    random_state=42
)
modelo.fit(X_train, y_train)

print(f"    Profundidad real del árbol : {modelo.get_depth()}")
print(f"    Número de hojas            : {modelo.get_n_leaves()}")

# ══════════════════════════════════════════════════════════════════
# 4. EVALUACIÓN DEL MODELO
# ══════════════════════════════════════════════════════════════════
print("\n[4] Evaluando modelo...")

y_pred = modelo.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n    ── Exactitud (Accuracy) ──────────────────────────")
print(f"    Entrenamiento : {modelo.score(X_train, y_train)*100:.2f}%")
print(f"    Prueba        : {acc*100:.2f}%")

# Validación cruzada (5-fold)
cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
print(f"\n    ── Validación Cruzada (5-fold) ────────────────────")
print(f"    Scores : {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Media  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

NOMBRES_CLASES = ['Rápida', 'Media', 'Larga']

print(f"\n    ── Reporte de Clasificación ───────────────────────")
print(classification_report(y_test, y_pred, target_names=NOMBRES_CLASES))

print(f"\n    ── Importancia de Features ────────────────────────")
importancias = pd.Series(modelo.feature_importances_, index=NOMBRES_FEATURES)
importancias = importancias.sort_values(ascending=False)
for feat, imp in importancias.items():
    barra = '█' * int(imp * 40)
    print(f"    {feat:20s}: {imp:.4f}  {barra}")

# ══════════════════════════════════════════════════════════════════
# 5. REGLAS EXTRAÍDAS DEL ÁRBOL (texto)
# ══════════════════════════════════════════════════════════════════
print("\n[5] Reglas del árbol (primeras 3 niveles):")
reglas = export_text(modelo, feature_names=NOMBRES_FEATURES, max_depth=3)
print(reglas)

# ══════════════════════════════════════════════════════════════════
# 6. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════
print("\n[6] Generando visualizaciones...")

# 6a. Árbol de decisión
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(modelo, feature_names=NOMBRES_FEATURES,
          class_names=NOMBRES_CLASES,
          filled=True, rounded=True, fontsize=8,
          impurity=True, proportion=False, ax=ax)
ax.set_title("Árbol de Decisión — Clasificación de Rutas TransMilenio\n"
             "Ref: Palma Méndez (2008), Cap. 17", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("arbol_decision.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → arbol_decision.png")

# 6b. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NOMBRES_CLASES)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Matriz de Confusión — Árbol de Decisión", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → matriz_confusion.png")

# 6c. Importancia de features
fig, ax = plt.subplots(figsize=(8, 4))
colores = ['#1F3864' if i == 0 else '#BDD7EE' for i in range(len(importancias))]
importancias.plot(kind='barh', ax=ax, color=colores[::-1])
ax.set_xlabel("Importancia (Gini)", fontsize=11)
ax.set_title("Importancia de Variables — Árbol de Decisión", fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
for i, v in enumerate(importancias.values[::-1]):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("importancia_features.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → importancia_features.png")

# ══════════════════════════════════════════════════════════════════
# 7. PREDICCIÓN DE EJEMPLO
# ══════════════════════════════════════════════════════════════════
print("\n[7] Predicción de ejemplo:")
print("    Consulta: Portal Norte → Portal Sur, hora pico mañana, día hábil")
print("    Troncal origen: Caracas(1), Troncal destino: Caracas(1)")
print("    Estaciones: 14, Transbordos: 0, Hora: 1 (mañana pico), Día: 0 (hábil)")

ejemplo = pd.DataFrame([[1, 1, 1, 0, 14, 0]], columns=FEATURES)
pred = modelo.predict(ejemplo)[0]
proba = modelo.predict_proba(ejemplo)[0]

print(f"\n    Resultado: {NOMBRES_CLASES[pred]}")
print(f"    Probabilidades: Rápida={proba[0]:.2%}  Media={proba[1]:.2%}  Larga={proba[2]:.2%}")

print("\n" + "=" * 65)
print("  ENTRENAMIENTO COMPLETADO")
print(f"  Exactitud final: {acc*100:.2f}%")
print("=" * 65)
