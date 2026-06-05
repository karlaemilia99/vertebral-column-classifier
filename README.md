# 🦴 Vertebral Column — Clasificación de Patologías Ortopédicas

Modelo de Machine Learning para clasificar pacientes en tres categorías diagnósticas — Hernia de Disco (DH), Normal (NO) y Espondilolistesis (SL) — a partir de 6 variables biomecánicas obtenidas de radiografías de columna y pelvis.

📊 [Ver reporte interactivo](reports/dashboard.html)

![Dashboard](reports/dashboard_static.png)

---

## 🩻 Contexto clínico

Los datos provienen de mediciones realizadas directamente sobre radiografías laterales de columna y pelvis. Cada paciente está representado por 6 variables:

| Variable | Descripción |
|---|---|
| `pelvic_incidence` | Ángulo entre la perpendicular al sacro y la línea al fémur — define la postura global |
| `pelvic_tilt` | Inclinación de la pelvis hacia adelante o atrás |
| `lumbar_lordosis_angle` | Curvatura natural de la zona lumbar |
| `sacral_slope` | Ángulo del sacro respecto a la horizontal |
| `pelvic_radius` | Distancia entre eje de caderas y extremo del sacro |
| `degree_spondylolisthesis` | Magnitud del desplazamiento vertebral — cercana a 0 en pacientes normales |

---

## 🧹 Preprocesamiento

### Carga del dataset
El archivo `.dat` no incluye encabezados — las columnas fueron asignadas manualmente al momento de la carga con `pd.read_csv(sep=' ', names=cols)`.

### Escalado
Se utilizó `StandardScaler` en lugar de `RobustScaler`. A diferencia de datasets financieros con outliers de millones, las variables biomecánicas tienen rangos naturalmente acotados. Con distribuciones relativamente normales, `StandardScaler` aprovecha mejor la información al usar media y desviación estándar completas.

### Split estratificado
Con solo 310 pacientes el split estratificado es crítico — garantiza que la proporción DH/NO/SL se mantenga en train y test. Sin estratificación, un split aleatorio podría dejar el test sin pacientes DH suficientes.

---

## 📊 Métricas — F1-score y validación cruzada

Con un dataset de 310 pacientes y distribución desbalanceada (48% SL, 32% NO, 19% DH), dos decisiones metodológicas fueron clave:

**F1-score weighted sobre accuracy**
Un modelo que predijera siempre SL obtendría 48% de accuracy sin haber aprendido nada. El F1-score evalúa cada clase por separado y pondera por su frecuencia real.

**Validación cruzada de 5 pliegues**
Con tan pocos pacientes, un único split puede ser afortunado o desfavorable. La validación cruzada evalúa el modelo 5 veces con particiones distintas — los resultados son más confiables que un solo test.

---

## 🏆 Resultados

| Modelo | F1 Test | CV Mean | CV Std |
|---|---|---|---|
| **Random Forest** | **0.8714** | 0.8273 | ±0.044 |
| Logistic Regression | 0.8287 | 0.8378 | ±0.061 |
| SVM | 0.8137 | 0.8243 | ±0.047 |
| KNN | 0.7237 | 0.7875 | ±0.009 |

### F1-score por clase — Random Forest

| Clase | Precisión | Recall | F1 |
|---|---|---|---|
| DH — Hernia de disco | 0.73 | 0.67 | 0.70 |
| NO — Normal | 0.77 | 0.85 | 0.81 |
| SL — Espondilolistesis | 1.00 | 0.97 | 0.98 |

### Observación clave
La espondilolistesis se detecta con F1 de 0.98 porque `degree_spondylolisthesis` la distingue inequívocamente — pacientes SL tienen desplazamientos de 0 a 400+, mientras DH y NO se mantienen cerca de 0.

La hernia de disco es el mayor reto (F1: 0.70) porque sus medidas biomecánicas se solapan con pacientes normales. Esto sugiere que variables adicionales — imagen directa, síntomas clínicos — podrían mejorar su detección.

---

## 🔍 Feature Importance

| Variable | Importancia |
|---|---|
| degree_spondylolisthesis | 34.7% |
| sacral_slope | 16.3% |
| pelvic_radius | 15.2% |
| pelvic_incidence | 14.8% |
| lumbar_lordosis_angle | 11.1% |
| pelvic_tilt | 7.9% |

`degree_spondylolisthesis` tiene el doble de importancia que cualquier otra variable — confirma que es la firma biomecánica definitoria de la espondilolistesis.

---

## 🗂️ Estructura del proyecto

```
vertebral-column-classifier/
├── data/
│   ├── column_3C.dat
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── models/
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── best_model.pkl
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   └── dashboard.py
├── reports/
│   └── dashboard.html
└── README.md
```

---

## ⚙️ Cómo ejecutar

### 1. Clona el repositorio
```bash
git clone https://github.com/tu-usuario/vertebral-column-classifier.git
cd vertebral-column-classifier
```

### 2. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecuta en orden
```bash
python src/eda.py
python src/preprocessing.py
python src/train.py
python src/dashboard.py
```

---

## 🛠️ Stack

- **Python 3.11**
- **pandas / numpy** — manipulación de datos
- **scikit-learn** — modelos, métricas, StandardScaler, cross_val_score
- **matplotlib / seaborn** — visualizaciones

---

## 📁 Dataset

[Vertebral Column — UCI ML Repository](https://archive.ics.uci.edu/dataset/212/vertebral+column)

310 pacientes ortopédicos con 6 variables biomecánicas. Construido por el Dr. Henrique da Mota durante residencia médica en el Centre Médico-Chirurgical de Réadaptation des Massues, Lyon, Francia.

---

## 👩‍💻 Autora

**Karla Altamirano** — Software Engineer & Digital Transformation Specialist  
[LinkedIn](https://www.linkedin.com/in/karlaemilia99/) · [GitHub](https://github.com/karlaemilia99)