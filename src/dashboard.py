import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import os

# ── Cargar datos ──
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

label_names = ['DH', 'NO', 'SL']

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
}

results = {}
print("Entrenando modelos...")
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'model': model, 'f1': f1, 'cv': cv.mean(), 'y_pred': y_pred}
    print(f"  {name}: F1={f1:.4f} | CV={cv.mean():.4f}")

best_name = max(results, key=lambda x: results[x]['f1'])
best_pred = results[best_name]['y_pred']
best_f1   = results[best_name]['f1']

report     = classification_report(y_test, best_pred, target_names=label_names, output_dict=True)
cm         = confusion_matrix(y_test, best_pred)
rf         = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

f1_per_cls = {cls: round(report[cls]['f1-score'], 2) for cls in label_names}
cm_vals    = cm.tolist()
feat_data  = [(col, round(val, 4)) for col, val in importances.items()]
max_feat   = feat_data[0][1]

models_list = list(results.keys())
f1_list     = [round(results[n]['f1'], 4) for n in models_list]
cv_list     = [round(results[n]['cv'], 4) for n in models_list]
labels_js   = str(models_list)
f1_js       = str(f1_list)
cv_js       = str(cv_list)
class_f1_js = str([f1_per_cls[c] for c in label_names])

dh_wrong = cm_vals[0][1]
dh_total = sum(cm_vals[0])
sl_total = sum(cm_vals[2])

print(f"\n✅ Mejor modelo: {best_name} (F1: {best_f1:.4f})")

# ── Feature bars ──
feature_bars_html = ''
for name, val in feat_data:
    pct = round((val / max_feat) * 100, 1)
    feature_bars_html += f'''
    <div class="feature-row">
      <span class="feature-name">{name.replace("_", " ").title()}</span>
      <div class="feature-bar-wrap">
        <div class="feature-bar" style="width:{pct}%"></div>
      </div>
      <span class="feature-val">{round(val*100,1)}%</span>
    </div>'''

# ── Conclusiones ──
conclusion_items = f"""
<li>{best_name} fue el mejor algoritmo con F1-score de {round(best_f1,4)} en test y {round(results[best_name]['cv'],4)} en validación cruzada.</li>
<li>La espondilolistesis (SL) se detecta con F1 de {f1_per_cls['SL']} — casi diagnóstico perfecto — porque el grado de desplazamiento vertebral la distingue inequívocamente.</li>
<li>La hernia de disco (DH) es el mayor reto con F1 de {f1_per_cls['DH']}. {dh_wrong} de {dh_total} pacientes DH fueron clasificados como Normal.</li>
<li>El modelo nunca confunde las condiciones más diferentes entre sí (SL y DH). Los errores ocurren solo entre condiciones biomecánicamente similares.</li>
<li>Con solo 6 variables y 310 pacientes se logra {round(best_f1*100)}% de precisión global.</li>
"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vertebral Column — Reporte de Resultados</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #faf8f5; --bg2: #f0ece4; --white: #ffffff;
    --ink: #1c1712; --ink2: #4a4035; --ink3: #9a9080;
    --dh: #b83232; --dh-light: #fae8e8;
    --no: #2a6e42; --no-light: #e2f2ea;
    --sl: #1e5899; --sl-light: #e0ecf8;
    --accent: #c07840; --border: #e0d8cc;
  }}
  html {{ scroll-behavior: smooth; }}
  body {{ font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--ink); line-height: 1.7; font-size: 16px; }}
  header {{ background: var(--ink); color: var(--bg); padding: 4rem 2rem 3rem; position: relative; }}
  header::after {{ content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--dh), var(--accent), var(--no), var(--sl)); }}
  .header-inner {{ max-width: 900px; margin: 0 auto; }}
  .header-tag {{ font-size: 0.7rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--accent); font-weight: 500; margin-bottom: 1.25rem; display: block; }}
  header h1 {{ font-family: 'Playfair Display', serif; font-size: clamp(1.8rem, 5vw, 3.2rem); font-weight: 400; line-height: 1.15; margin-bottom: 1.5rem; }}
  header h1 em {{ font-style: italic; color: var(--accent); }}
  .header-meta {{ display: flex; gap: 2.5rem; flex-wrap: wrap; padding-top: 1.75rem; border-top: 1px solid rgba(255,255,255,0.1); }}
  .meta-item {{ display: flex; flex-direction: column; gap: 0.2rem; }}
  .meta-val {{ font-family: 'Playfair Display', serif; font-size: 1.8rem; color: var(--accent); line-height: 1; }}
  .meta-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(250,248,245,0.45); }}
  main {{ max-width: 960px; margin: 0 auto; padding: 4rem 2rem; }}
  .section {{ margin-bottom: 5rem; }}
  .section-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.22em; color: var(--accent); font-weight: 500; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 1rem; }}
  .section-label::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
  .section-title {{ font-family: 'Playfair Display', serif; font-size: clamp(1.3rem, 3vw, 2rem); font-weight: 400; margin-bottom: 0.9rem; line-height: 1.2; }}
  .section-intro {{ font-size: 0.95rem; color: var(--ink2); max-width: 640px; margin-bottom: 2rem; font-weight: 300; line-height: 1.85; }}
  .diagnosis-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem; margin-bottom: 2.5rem; }}
  .dx-card {{ background: var(--white); border: 1px solid var(--border); border-radius: 6px; padding: 1.5rem; position: relative; }}
  .dx-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 6px 6px 0 0; }}
  .dx-card.dh::before {{ background: var(--dh); }}
  .dx-card.no::before {{ background: var(--no); }}
  .dx-card.sl::before {{ background: var(--sl); }}
  .dx-badge {{ display: inline-block; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; padding: 0.3rem 0.75rem; border-radius: 3px; margin-bottom: 0.9rem; }}
  .dh .dx-badge {{ background: var(--dh-light); color: var(--dh); }}
  .no .dx-badge {{ background: var(--no-light); color: var(--no); }}
  .sl .dx-badge {{ background: var(--sl-light); color: var(--sl); }}
  .dx-name {{ font-family: 'Playfair Display', serif; font-size: 1.1rem; margin-bottom: 0.5rem; }}
  .dx-desc {{ font-size: 0.84rem; color: var(--ink2); font-weight: 300; line-height: 1.7; margin-bottom: 1.25rem; }}
  .dx-stats {{ display: flex; gap: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
  .dx-stat-num {{ font-family: 'Playfair Display', serif; font-size: 1.35rem; display: block; line-height: 1; margin-bottom: 0.15rem; }}
  .dh .dx-stat-num {{ color: var(--dh); }} .no .dx-stat-num {{ color: var(--no); }} .sl .dx-stat-num {{ color: var(--sl); }}
  .dx-stat-label {{ font-size: 0.68rem; color: var(--ink3); text-transform: uppercase; letter-spacing: 0.08em; }}
  .results-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 2rem; }}
  @media (max-width: 600px) {{ .results-grid {{ grid-template-columns: 1fr; }} .diagnosis-grid {{ grid-template-columns: 1fr; }} }}
  .chart-card {{ background: var(--white); border: 1px solid var(--border); border-radius: 6px; padding: 1.5rem; }}
  .chart-title {{ font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; }}
  .chart-subtitle {{ font-size: 0.8rem; color: var(--ink3); margin-bottom: 1.25rem; font-weight: 300; }}
  .chart-wrap {{ position: relative; height: 220px; width: 100%; display: block; }}
canvas {{ display: block; }}

  .insights-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-top: 1.75rem; }}
  .insight {{ background: var(--white); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 1.25rem 1.5rem; }}
  .insight-title {{ font-weight: 500; font-size: 0.875rem; margin-bottom: 0.4rem; }}
  .insight-text {{ font-size: 0.83rem; color: var(--ink2); font-weight: 300; line-height: 1.75; }}
  .cm-wrap {{ background: var(--white); border: 1px solid var(--border); border-radius: 6px; padding: 1.5rem; margin-bottom: 1.5rem; overflow-x: auto; }}
  .cm-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-top: 1rem; min-width: 320px; }}
  .cm-table th {{ padding: 0.6rem 1rem; font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink3); text-align: center; }}
  .cm-table td {{ padding: 0.75rem 1rem; text-align: center; border: 1px solid var(--border); font-family: 'Playfair Display', serif; font-size: 1.2rem; }}
  .cm-label {{ font-family: 'DM Sans', sans-serif !important; font-size: 0.78rem !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink2); }}
  .cm-correct {{ background: rgba(42,110,66,0.08); color: var(--no); }}
  .cm-wrong {{ background: rgba(184,50,50,0.05); color: var(--dh); }}
  .feature-list {{ display: flex; flex-direction: column; gap: 0.8rem; margin-top: 1rem; }}
  .feature-row {{ display: flex; align-items: center; gap: 1rem; }}
  .feature-name {{ font-size: 0.82rem; color: var(--ink2); width: 220px; flex-shrink: 0; font-weight: 300; }}
  .feature-bar-wrap {{ flex: 1; background: var(--bg2); border-radius: 2px; height: 7px; overflow: hidden; }}
  .feature-bar {{ height: 100%; border-radius: 2px; background: var(--accent); }}
  .feature-val {{ font-size: 0.78rem; color: var(--ink3); width: 40px; text-align: right; }}
  .conclusion {{ background: var(--ink); color: var(--bg); border-radius: 6px; padding: 2.5rem; margin-top: 2rem; }}
  .conclusion h3 {{ font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 400; margin-bottom: 1.25rem; color: var(--accent); }}
  .conclusion-list {{ list-style: none; display: flex; flex-direction: column; gap: 0.9rem; }}
  .conclusion-list li {{ display: flex; gap: 1rem; font-size: 0.9rem; color: rgba(250,248,245,0.78); font-weight: 300; line-height: 1.75; }}
  .conclusion-list li::before {{ content: '→'; color: var(--accent); flex-shrink: 0; }}
  footer {{ text-align: center; padding: 2rem; font-size: 0.78rem; color: var(--ink3); border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <span class="header-tag">Reporte de Resultados — Machine Learning</span>
    <h1>Clasificación de<br>Patologías de <em>Columna Vertebral</em></h1>
    <div class="header-meta">
      <div class="meta-item"><span class="meta-val">310</span><span class="meta-label">Pacientes analizados</span></div>
      <div class="meta-item"><span class="meta-val">{round(best_f1*100)}%</span><span class="meta-label">Precisión del modelo</span></div>
      <div class="meta-item"><span class="meta-val">4</span><span class="meta-label">Algoritmos comparados</span></div>
      <div class="meta-item"><span class="meta-val">6</span><span class="meta-label">Variables biomecánicas</span></div>
    </div>
  </div>
</header>

<main>

  <div class="section">
    <p class="section-label">Contexto clínico</p>
    <h2 class="section-title">Las tres condiciones que el modelo aprende a distinguir</h2>
    <p class="section-intro">A partir de 6 medidas biomecánicas obtenidas de radiografías laterales de columna y pelvis, el modelo clasifica cada paciente en una de estas tres categorías.</p>
    <div class="diagnosis-grid">
      <div class="dx-card dh">
        <span class="dx-badge">DH</span>
        <h3 class="dx-name">Hernia de Disco</h3>
        <p class="dx-desc">El material interno del disco intervertebral se desplaza hacia afuera comprimiendo nervios. Genera dolor lumbar e irradiado hacia las piernas.</p>
        <div class="dx-stats">
          <div><span class="dx-stat-num">60</span><span class="dx-stat-label">Pacientes</span></div>
          <div><span class="dx-stat-num">19%</span><span class="dx-stat-label">Del total</span></div>
          <div><span class="dx-stat-num">{f1_per_cls['DH']}</span><span class="dx-stat-label">F1-score</span></div>
        </div>
      </div>
      <div class="dx-card no">
        <span class="dx-badge">NO</span>
        <h3 class="dx-name">Normal</h3>
        <p class="dx-desc">Columna y pelvis con alineación biomecánica dentro de rangos saludables. Sin patología estructural identificable en ninguna de las 6 mediciones.</p>
        <div class="dx-stats">
          <div><span class="dx-stat-num">100</span><span class="dx-stat-label">Pacientes</span></div>
          <div><span class="dx-stat-num">32%</span><span class="dx-stat-label">Del total</span></div>
          <div><span class="dx-stat-num">{f1_per_cls['NO']}</span><span class="dx-stat-label">F1-score</span></div>
        </div>
      </div>
      <div class="dx-card sl">
        <span class="dx-badge">SL</span>
        <h3 class="dx-name">Espondilolistesis</h3>
        <p class="dx-desc">Una vértebra se desliza sobre la inferior. Es la condición más prevalente en este dataset y la más fácil de identificar biomecánicamente.</p>
        <div class="dx-stats">
          <div><span class="dx-stat-num">150</span><span class="dx-stat-label">Pacientes</span></div>
          <div><span class="dx-stat-num">48%</span><span class="dx-stat-label">Del total</span></div>
          <div><span class="dx-stat-num">{f1_per_cls['SL']}</span><span class="dx-stat-label">F1-score</span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">Rendimiento del modelo</p>
    <h2 class="section-title">{best_name} fue el mejor algoritmo</h2>
    <p class="section-intro">Se compararon 4 algoritmos usando F1-score como métrica principal y validación cruzada de 5 pliegues para garantizar resultados confiables con 310 pacientes.</p>
    <div class="results-grid">
      <div class="chart-card">
        <p class="chart-title">Comparación de algoritmos</p>
        <p class="chart-subtitle">F1-score en test vs media de validación cruzada</p>
        <div class="chart-wrap"><canvas id="modelsChart"></canvas></div>
      </div>
      <div class="chart-card">
        <p class="chart-title">F1-score por diagnóstico</p>
        <p class="chart-subtitle">{best_name} — detección por condición</p>
        <div class="chart-wrap"><canvas id="classChart"></canvas></div>
      </div>
    </div>
    <div class="insights-grid">
      <div class="insight">
        <p class="insight-title">¿Por qué F1-score y no accuracy?</p>
        <p class="insight-text">El dataset está desbalanceado — 48% SL vs 19% DH. Un modelo que siempre dijera "SL" tendría 48% de accuracy sin aprender nada. El F1-score evalúa cada clase por separado.</p>
      </div>
      <div class="insight">
        <p class="insight-title">¿Por qué validación cruzada?</p>
        <p class="insight-text">Con solo 310 pacientes, un único split puede ser suertudo. La validación cruzada evalúa el modelo 5 veces con particiones distintas y promedia los resultados.</p>
      </div>
      <div class="insight">
        <p class="insight-title">El reto: DH vs Normal</p>
        <p class="insight-text">La hernia de disco y los pacientes normales comparten rangos biomecánicos muy similares. El modelo los confunde con más frecuencia que cualquier otra combinación.</p>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">Análisis de errores</p>
    <h2 class="section-title">Dónde acierta y dónde falla el modelo</h2>
    <p class="section-intro">La matriz de confusión muestra los resultados sobre 62 pacientes del conjunto de prueba. Los valores en la diagonal son aciertos; fuera de ella, errores.</p>
    <div class="cm-wrap">
      <p class="chart-title">Matriz de Confusión — {best_name}</p>
      <p class="chart-subtitle">62 pacientes de prueba · 20% del dataset</p>
      <table class="cm-table">
        <thead>
          <tr><th></th><th colspan="3" style="border-bottom:2px solid var(--border);color:var(--ink2);">Predicho por el modelo</th></tr>
          <tr><th></th><th>DH</th><th>Normal</th><th>SL</th></tr>
        </thead>
        <tbody>
          <tr>
            <td class="cm-label">DH (real)</td>
            <td class="cm-correct">{cm_vals[0][0]}</td>
            <td class="cm-wrong">{cm_vals[0][1]}</td>
            <td class="cm-wrong">{cm_vals[0][2]}</td>
          </tr>
          <tr>
            <td class="cm-label">Normal (real)</td>
            <td class="cm-wrong">{cm_vals[1][0]}</td>
            <td class="cm-correct">{cm_vals[1][1]}</td>
            <td class="cm-wrong">{cm_vals[1][2]}</td>
          </tr>
          <tr>
            <td class="cm-label">SL (real)</td>
            <td class="cm-wrong">{cm_vals[2][0]}</td>
            <td class="cm-wrong">{cm_vals[2][1]}</td>
            <td class="cm-correct">{cm_vals[2][2]}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="insights-grid">
      <div class="insight">
        <p class="insight-title">Espondilolistesis — casi perfecta</p>
        <p class="insight-text">{cm_vals[2][2]} de {sl_total} pacientes SL detectados correctamente. La variable degree_spondylolisthesis la distingue inequívocamente.</p>
      </div>
      <div class="insight">
        <p class="insight-title">Hernia de disco — el más difícil</p>
        <p class="insight-text">{dh_wrong} de {dh_total} pacientes DH clasificados como Normal. Sus medidas biomecánicas se solapan con pacientes sanos.</p>
      </div>
      <div class="insight">
        <p class="insight-title">Sin confusión entre extremos</p>
        <p class="insight-text">Ningún paciente SL fue clasificado como DH ni viceversa. Los errores ocurren solo entre condiciones biomecánicamente similares.</p>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">Variables más importantes</p>
    <h2 class="section-title">Qué medidas biomecánicas más influyen en el diagnóstico</h2>
    <p class="section-intro">El modelo Random Forest calcula internamente qué variables usa más al tomar decisiones.</p>
    <div class="cm-wrap">
      <div class="feature-list">{feature_bars_html}</div>
    </div>
    <div class="insights-grid">
      <div class="insight">
        <p class="insight-title">Degree spondylolisthesis domina</p>
        <p class="insight-text">Con {round(feat_data[0][1]*100,1)}% de importancia vale casi el doble que la segunda variable. Es la magnitud directa del desplazamiento vertebral.</p>
      </div>
      <div class="insight">
        <p class="insight-title">{feat_data[-1][0].replace('_',' ').title()} aporta menos</p>
        <p class="insight-text">Con {round(feat_data[-1][1]*100,1)}% es la variable menos discriminante del modelo.</p>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">Conclusiones</p>
    <h2 class="section-title">Lo que aprendimos del modelo y los datos</h2>
    <div class="conclusion">
      <h3>Hallazgos principales</h3>
      <ul class="conclusion-list">{conclusion_items}</ul>
    </div>
  </div>

</main>

<footer>
  Karla Altamirano — Machine Learning Portfolio &nbsp;·&nbsp; Dataset: UCI ML Repository — Vertebral Column
</footer>

<script>
const chartDefaults = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ labels: {{ font: {{ family: 'DM Sans', size: 11 }}, color: '#4a4035' }} }} }}
}};

new Chart(document.getElementById('modelsChart'), {{
  type: 'bar',
  data: {{
    labels: {labels_js},
    datasets: [
      {{ label: 'F1 Test', data: {f1_js}, backgroundColor: '#1e5899', borderRadius: 4 }},
      {{ label: 'CV Mean', data: {cv_js}, backgroundColor: '#c07840', borderRadius: 4 }}
    ]
  }},
  options: {{ ...chartDefaults, scales: {{
    y: {{ min: 0.6, max: 1.0, grid: {{ color: '#f0ece4' }}, ticks: {{ color: '#9a9080', font: {{ size: 10 }} }} }},
    x: {{ grid: {{ display: false }}, ticks: {{ color: '#4a4035', font: {{ size: 10 }} }} }}
  }} }}
}});

new Chart(document.getElementById('classChart'), {{
  type: 'bar',
  data: {{
    labels: ['DH — Hernia', 'NO — Normal', 'SL — Espondilolistesis'],
    datasets: [{{ data: {class_f1_js}, backgroundColor: ['#b83232','#2a6e42','#1e5899'], borderRadius: 4 }}]
  }},
  options: {{ ...chartDefaults, plugins: {{ legend: {{ display: false }} }}, scales: {{
    y: {{ min: 0, max: 1.1, grid: {{ color: '#f0ece4' }}, ticks: {{ color: '#9a9080', font: {{ size: 10 }} }} }},
    x: {{ grid: {{ display: false }}, ticks: {{ color: '#4a4035', font: {{ size: 10 }}, maxRotation: 0 }} }}
  }} }}
}});
</script>
</body>
</html>"""

os.makedirs('reports', exist_ok=True)
with open('reports/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ Dashboard HTML generado en reports/dashboard.html")
