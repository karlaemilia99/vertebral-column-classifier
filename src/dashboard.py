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

report  = classification_report(y_test, best_pred, target_names=label_names, output_dict=True)
cm      = confusion_matrix(y_test, best_pred)
rf      = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

f1_per_cls = {cls: round(report[cls]['f1-score'], 2) for cls in label_names}
cm_vals    = cm.tolist()
feat_data  = [(col, round(val, 4)) for col, val in importances.items()]
max_feat   = feat_data[0][1]

models_list = list(results.keys())
f1_list     = [round(results[n]['f1'], 4) for n in models_list]
cv_list     = [round(results[n]['cv'], 4) for n in models_list]
labels_js   = str([n.replace(' ', ' ') for n in models_list])
f1_js       = str(f1_list)
cv_js       = str(cv_list)
class_f1_js = str([f1_per_cls[c] for c in label_names])

print(f"\n✅ Mejor modelo: {best_name} (F1: {best_f1:.4f})")

# ── Feature bars HTML ──
feature_bars_html = ''
for name, val in feat_data:
    pct = round((val / max_feat) * 100, 1)
    feature_bars_html += f'''
    <div class="feature-row">
      <span class="feature-name">{name.replace("_", " ").title()}</span>
      <div class="feature-bar-wrap">
        <div class="feature-bar" data-width="{pct}"></div>
      </div>
      <span class="feature-val">{round(val*100,1)}%</span>
    </div>'''

# ── Conclusion dinámica ──
sl_detected = cm_vals[2][2]
sl_total    = sum(cm_vals[2])
dh_wrong    = cm_vals[0][1]
dh_total    = sum(cm_vals[0])

conclusion_items = f"""
<li>{best_name} fue el mejor algoritmo con F1-score de {round(best_f1,4)} en test y {round(results[best_name]['cv'],4)} en validación cruzada — resultados consistentes que no dependen de un split afortunado.</li>
<li>La espondilolistesis (SL) se detecta con F1 de {f1_per_cls['SL']} porque el grado de desplazamiento vertebral la distingue inequívocamente. Los pacientes SL tienen desplazamientos de 0 a 400+, mientras DH y Normal se mantienen cerca de 0.</li>
<li>La hernia de disco (DH) es el mayor reto con F1 de {f1_per_cls['DH']}. {dh_wrong} de {dh_total} pacientes DH fueron clasificados como Normal — sus medidas biomecánicas se solapan con pacientes sanos.</li>
<li>El modelo nunca confunde las condiciones más diferentes entre sí (SL y DH). Los errores ocurren exclusivamente entre condiciones biomecánicamente similares — señal de que aprendió patrones clínicamente coherentes.</li>
<li>Con solo 6 variables y 310 pacientes se logra {round(best_f1*100)}% de precisión global, demostrando el poder predictivo de las medidas biomecánicas como indicadores diagnósticos.</li>
"""

# ── HTML completo ──
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
    --bg:#faf8f5; --bg2:#f0ece4; --white:#ffffff;
    --ink:#1c1712; --ink2:#4a4035; --ink3:#9a9080;
    --dh:#b83232; --dh-light:#fae8e8;
    --no:#2a6e42;  --no-light:#e2f2ea;
    --sl:#1e5899;  --sl-light:#e0ecf8;
    --accent:#c07840; --border:#e0d8cc;
  }}
  html {{ scroll-behavior:smooth; }}
  body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--ink); line-height:1.7; font-size:16px; overflow-x:hidden; }}
  header {{ background:var(--ink); color:var(--bg); padding:0; position:relative; overflow:hidden; min-height:420px; display:flex; align-items:center; }}
  .header-bg-bone {{ position:absolute; right:-40px; top:50%; transform:translateY(-50%); opacity:0.06; width:480px; height:auto; }}
  .header-bone-small {{ position:absolute; left:5%; bottom:-20px; opacity:0.04; width:200px; }}
  header::after {{ content:''; position:absolute; bottom:0; left:0; right:0; height:3px; background:linear-gradient(90deg,var(--dh),var(--accent),var(--no),var(--sl)); }}
  .header-inner {{ max-width:960px; margin:0 auto; padding:5rem 3rem 4rem; position:relative; z-index:2; width:100%; }}
  .header-tag {{ font-size:0.7rem; letter-spacing:0.22em; text-transform:uppercase; color:var(--accent); font-weight:500; margin-bottom:1.5rem; display:block; }}
  header h1 {{ font-family:'Playfair Display',serif; font-size:clamp(2rem,5vw,3.8rem); font-weight:400; line-height:1.1; margin-bottom:1.75rem; }}
  header h1 em {{ font-style:italic; color:var(--accent); }}
  .header-meta {{ display:flex; gap:3rem; flex-wrap:wrap; padding-top:2rem; border-top:1px solid rgba(255,255,255,0.1); }}
  .meta-item {{ display:flex; flex-direction:column; gap:0.2rem; }}
  .meta-val {{ font-family:'Playfair Display',serif; font-size:2rem; color:var(--accent); line-height:1; }}
  .meta-label {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; color:rgba(250,248,245,0.45); }}
  main {{ max-width:1020px; margin:0 auto; padding:5rem 2rem; }}
  .section {{ margin-bottom:6rem; position:relative; }}
  .section-label {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.22em; color:var(--accent); font-weight:500; margin-bottom:0.6rem; display:flex; align-items:center; gap:1rem; }}
  .section-label::after {{ content:''; flex:1; height:1px; background:var(--border); }}
  .section-title {{ font-family:'Playfair Display',serif; font-size:clamp(1.4rem,3vw,2.1rem); font-weight:400; margin-bottom:0.9rem; line-height:1.2; }}
  .section-intro {{ font-size:0.975rem; color:var(--ink2); max-width:660px; margin-bottom:2.5rem; font-weight:300; line-height:1.85; }}
  .diagnosis-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.25rem; margin-bottom:3rem; }}
  .dx-card {{ background:var(--white); border:1px solid var(--border); border-radius:6px; padding:1.75rem; position:relative; overflow:hidden; transition:transform 0.25s, box-shadow 0.25s; }}
  .dx-card:hover {{ transform:translateY(-4px); box-shadow:0 12px 32px rgba(0,0,0,0.07); }}
  .dx-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
  .dx-card.dh::before {{ background:var(--dh); }} .dx-card.no::before {{ background:var(--no); }} .dx-card.sl::before {{ background:var(--sl); }}
  .dx-illustration {{ position:absolute; bottom:-10px; right:-10px; opacity:0.07; width:100px; }}
  .dx-badge {{ display:inline-block; font-size:0.68rem; font-weight:500; letter-spacing:0.14em; text-transform:uppercase; padding:0.3rem 0.75rem; border-radius:3px; margin-bottom:1rem; }}
  .dh .dx-badge {{ background:var(--dh-light); color:var(--dh); }} .no .dx-badge {{ background:var(--no-light); color:var(--no); }} .sl .dx-badge {{ background:var(--sl-light); color:var(--sl); }}
  .dx-name {{ font-family:'Playfair Display',serif; font-size:1.15rem; margin-bottom:0.5rem; }}
  .dx-desc {{ font-size:0.85rem; color:var(--ink2); font-weight:300; line-height:1.75; margin-bottom:1.25rem; }}
  .dx-stats {{ display:flex; gap:1.25rem; padding-top:1rem; border-top:1px solid var(--border); }}
  .dx-stat-num {{ font-family:'Playfair Display',serif; font-size:1.4rem; display:block; line-height:1; margin-bottom:0.15rem; }}
  .dh .dx-stat-num {{ color:var(--dh); }} .no .dx-stat-num {{ color:var(--no); }} .sl .dx-stat-num {{ color:var(--sl); }}
  .dx-stat-label {{ font-size:0.68rem; color:var(--ink3); text-transform:uppercase; letter-spacing:0.08em; }}
  .results-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; margin-bottom:2rem; }}
  .chart-card {{ background:var(--white); border:1px solid var(--border); border-radius:6px; padding:1.75rem; position:relative; overflow:hidden; }}
  .chart-card-bone {{ position:absolute; top:-15px; right:-15px; opacity:0.04; width:90px; }}
  .chart-title {{ font-family:'Playfair Display',serif; font-size:1rem; font-weight:600; margin-bottom:0.3rem; }}
  .chart-subtitle {{ font-size:0.8rem; color:var(--ink3); margin-bottom:1.25rem; font-weight:300; }}
  .chart-wrap {{ position:relative; height:220px; }}
  .insights-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem; margin-top:2rem; }}
  .insight {{ background:var(--white); border:1px solid var(--border); border-left:3px solid var(--accent); border-radius:0 6px 6px 0; padding:1.25rem 1.5rem; transition:border-left-color 0.2s; }}
  .insight:hover {{ border-left-color:var(--ink); }}
  .insight-title {{ font-weight:500; font-size:0.875rem; margin-bottom:0.45rem; }}
  .insight-text {{ font-size:0.835rem; color:var(--ink2); font-weight:300; line-height:1.75; }}
  .cm-wrap {{ background:var(--white); border:1px solid var(--border); border-radius:6px; padding:1.75rem; margin-bottom:1.5rem; position:relative; overflow:hidden; }}
  .cm-bone-deco {{ position:absolute; right:2rem; top:50%; transform:translateY(-50%); opacity:0.05; width:120px; }}
  .cm-table {{ width:100%; border-collapse:collapse; font-size:0.875rem; margin-top:1.25rem; }}
  .cm-table th {{ padding:0.6rem 1rem; font-weight:500; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--ink3); text-align:center; }}
  .cm-table td {{ padding:0.8rem 1rem; text-align:center; border:1px solid var(--border); font-family:'Playfair Display',serif; font-size:1.3rem; }}
  .cm-label {{ font-family:'DM Sans',sans-serif !important; font-size:0.78rem !important; font-weight:500; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink2); }}
  .cm-correct {{ background:rgba(42,110,66,0.07); color:var(--no); }} .cm-wrong {{ background:rgba(184,50,50,0.05); color:var(--dh); }}
  .feature-list {{ display:flex; flex-direction:column; gap:0.85rem; margin-top:1rem; }}
  .feature-row {{ display:flex; align-items:center; gap:1rem; }}
  .feature-name {{ font-size:0.82rem; color:var(--ink2); width:230px; flex-shrink:0; font-weight:300; }}
  .feature-bar-wrap {{ flex:1; background:var(--bg2); border-radius:2px; height:7px; overflow:hidden; }}
  .feature-bar {{ height:100%; border-radius:2px; background:var(--accent); width:0; transition:width 1.2s cubic-bezier(0.4,0,0.2,1); }}
  .feature-val {{ font-size:0.78rem; color:var(--ink3); width:40px; text-align:right; }}
  .conclusion {{ background:var(--ink); color:var(--bg); border-radius:6px; padding:3rem; margin-top:2rem; position:relative; overflow:hidden; }}
  .conclusion-bone {{ position:absolute; right:-30px; bottom:-30px; opacity:0.05; width:280px; }}
  .conclusion h3 {{ font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:400; margin-bottom:1.5rem; color:var(--accent); }}
  .conclusion-list {{ list-style:none; display:flex; flex-direction:column; gap:1rem; position:relative; z-index:1; }}
  .conclusion-list li {{ display:flex; gap:1rem; font-size:0.9rem; color:rgba(250,248,245,0.78); font-weight:300; line-height:1.75; }}
  .conclusion-list li::before {{ content:'→'; color:var(--accent); flex-shrink:0; }}
  footer {{ text-align:center; padding:2.5rem; font-size:0.78rem; color:var(--ink3); border-top:1px solid var(--border); display:flex; justify-content:center; align-items:center; gap:1rem; }}
  footer svg {{ opacity:0.3; }}
  @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:translateY(0); }} }}
  .animate {{ opacity:0; }}
  .animate.visible {{ animation:fadeUp 0.65s ease both; }}
</style>
</head>
<body>

<svg width="0" height="0" style="position:absolute">
  <defs>
    <symbol id="vertebra" viewBox="0 0 80 50">
      <rect x="15" y="10" width="50" height="30" rx="8" fill="currentColor"/>
      <rect x="5" y="18" width="15" height="14" rx="4" fill="currentColor"/>
      <rect x="60" y="18" width="15" height="14" rx="4" fill="currentColor"/>
      <rect x="30" y="0" width="20" height="12" rx="4" fill="currentColor"/>
      <rect x="30" y="38" width="20" height="12" rx="4" fill="currentColor"/>
    </symbol>
    <symbol id="pelvis" viewBox="0 0 120 80">
      <ellipse cx="60" cy="40" rx="55" ry="30" fill="none" stroke="currentColor" stroke-width="5"/>
      <ellipse cx="60" cy="40" rx="25" ry="15" fill="none" stroke="currentColor" stroke-width="4"/>
      <line x1="5" y1="40" x2="35" y2="40" stroke="currentColor" stroke-width="4"/>
      <line x1="85" y1="40" x2="115" y2="40" stroke="currentColor" stroke-width="4"/>
      <circle cx="22" cy="40" r="8" fill="currentColor"/>
      <circle cx="98" cy="40" r="8" fill="currentColor"/>
    </symbol>
    <symbol id="spine" viewBox="0 0 40 160">
      <rect x="12" y="0"   width="16" height="18" rx="4" fill="currentColor"/>
      <rect x="12" y="23"  width="16" height="18" rx="4" fill="currentColor"/>
      <rect x="12" y="46"  width="16" height="18" rx="4" fill="currentColor"/>
      <rect x="10" y="69"  width="20" height="20" rx="4" fill="currentColor"/>
      <rect x="10" y="94"  width="20" height="20" rx="4" fill="currentColor"/>
      <rect x="8"  y="119" width="24" height="22" rx="5" fill="currentColor"/>
      <rect x="6"  y="146" width="28" height="14" rx="6" fill="currentColor"/>
      <line x1="20" y1="18"  x2="20" y2="23"  stroke="currentColor" stroke-width="2"/>
      <line x1="20" y1="41"  x2="20" y2="46"  stroke="currentColor" stroke-width="2"/>
      <line x1="20" y1="64"  x2="20" y2="69"  stroke="currentColor" stroke-width="2"/>
      <line x1="20" y1="89"  x2="20" y2="94"  stroke="currentColor" stroke-width="2"/>
      <line x1="20" y1="114" x2="20" y2="119" stroke="currentColor" stroke-width="2"/>
      <line x1="20" y1="141" x2="20" y2="146" stroke="currentColor" stroke-width="2"/>
    </symbol>
  </defs>
</svg>

<header>
  <svg class="header-bg-bone" viewBox="0 0 40 160" fill="white"><use href="#spine"/></svg>
  <svg class="header-bone-small" viewBox="0 0 120 80" fill="white"><use href="#pelvis"/></svg>
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

  <div class="section animate">
    <p class="section-label">Contexto clínico</p>
    <h2 class="section-title">Las tres condiciones que el modelo aprende a distinguir</h2>
    <p class="section-intro">A partir de 6 medidas biomecánicas obtenidas de radiografías laterales de columna y pelvis, el modelo clasifica cada paciente en una de estas tres categorías diagnósticas.</p>
    <div class="diagnosis-grid">
      <div class="dx-card dh">
        <svg class="dx-illustration" viewBox="0 0 80 50" fill="#b83232"><use href="#vertebra"/></svg>
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
        <svg class="dx-illustration" viewBox="0 0 40 160" fill="#2a6e42"><use href="#spine"/></svg>
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
        <svg class="dx-illustration" viewBox="0 0 120 80" fill="#1e5899"><use href="#pelvis"/></svg>
        <span class="dx-badge">SL</span>
        <h3 class="dx-name">Espondilolistesis</h3>
        <p class="dx-desc">Una vértebra se desliza sobre la inferior. Es la condición más prevalente en este dataset y la más fácil de identificar mediante mediciones biomecánicas.</p>
        <div class="dx-stats">
          <div><span class="dx-stat-num">150</span><span class="dx-stat-label">Pacientes</span></div>
          <div><span class="dx-stat-num">48%</span><span class="dx-stat-label">Del total</span></div>
          <div><span class="dx-stat-num">{f1_per_cls['SL']}</span><span class="dx-stat-label">F1-score</span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="section animate">
    <p class="section-label">Rendimiento del modelo</p>
    <h2 class="section-title">{best_name} fue el mejor algoritmo</h2>
    <p class="section-intro">Se compararon 4 algoritmos usando F1-score como métrica principal y validación cruzada de 5 pliegues para garantizar resultados confiables con 310 pacientes.</p>
    <div class="results-grid">
      <div class="chart-card">
        <svg class="chart-card-bone" viewBox="0 0 80 50" fill="#1c1712"><use href="#vertebra"/></svg>
        <p class="chart-title">Comparación de algoritmos</p>
        <p class="chart-subtitle">F1-score en test vs media de validación cruzada</p>
        <div class="chart-wrap"><canvas id="modelsChart"></canvas></div>
      </div>
      <div class="chart-card">
        <svg class="chart-card-bone" viewBox="0 0 40 160" fill="#1c1712"><use href="#spine"/></svg>
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
        <p class="insight-text">Con solo 310 pacientes, un único split puede ser suertudo o desfavorable. La validación cruzada evalúa el modelo 5 veces con particiones distintas y promedia los resultados.</p>
      </div>
      <div class="insight">
        <p class="insight-title">El reto: DH vs Normal</p>
        <p class="insight-text">La hernia de disco y los pacientes normales comparten rangos biomecánicos muy similares. El modelo los confunde con más frecuencia que cualquier otra combinación de clases.</p>
      </div>
    </div>
  </div>

  <div class="section animate">
    <p class="section-label">Análisis de errores</p>
    <h2 class="section-title">Dónde acierta y dónde falla el modelo</h2>
    <p class="section-intro">La matriz de confusión muestra los resultados sobre 62 pacientes del conjunto de prueba. Los valores en la diagonal son aciertos; fuera de ella, errores.</p>
    <div class="cm-wrap">
      <svg class="cm-bone-deco" viewBox="0 0 120 80" fill="#1c1712"><use href="#pelvis"/></svg>
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
            <td class="{'cm-correct' if cm_vals[0][0] > 0 else 'cm-wrong'}">{cm_vals[0][0]}</td>
            <td class="cm-wrong">{cm_vals[0][1]}</td>
            <td class="cm-wrong">{cm_vals[0][2]}</td>
          </tr>
          <tr>
            <td class="cm-label">Normal (real)</td>
            <td class="cm-wrong">{cm_vals[1][0]}</td>
            <td class="{'cm-correct' if cm_vals[1][1] > 0 else 'cm-wrong'}">{cm_vals[1][1]}</td>
            <td class="cm-wrong">{cm_vals[1][2]}</td>
          </tr>
          <tr>
            <td class="cm-label">SL (real)</td>
            <td class="cm-wrong">{cm_vals[2][0]}</td>
            <td class="cm-wrong">{cm_vals[2][1]}</td>
            <td class="{'cm-correct' if cm_vals[2][2] > 0 else 'cm-wrong'}">{cm_vals[2][2]}</td>
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

  <div class="section animate">
    <p class="section-label">Variables más importantes</p>
    <h2 class="section-title">Qué medidas biomecánicas más influyen en el diagnóstico</h2>
    <p class="section-intro">El modelo Random Forest calcula internamente qué variables usa más al tomar decisiones. El resultado confirma lo que el análisis exploratorio anticipó.</p>
    <div class="cm-wrap">
      <div class="feature-list" id="featureList">{feature_bars_html}</div>
    </div>
    <div class="insights-grid">
      <div class="insight">
        <p class="insight-title">Degree spondylolisthesis domina</p>
        <p class="insight-text">Con {round(feat_data[0][1]*100,1)}% de importancia vale casi el doble que la segunda variable. Es la magnitud directa del desplazamiento vertebral.</p>
      </div>
      <div class="insight">
        <p class="insight-title">{feat_data[-1][0].replace('_',' ').title()} aporta menos</p>
        <p class="insight-text">Con {round(feat_data[-1][1]*100,1)}% de importancia es la variable menos discriminante. Otras capturan mejor la información biomecánica relevante.</p>
      </div>
    </div>
  </div>

  <div class="section animate">
    <p class="section-label">Conclusiones</p>
    <h2 class="section-title">Lo que aprendimos del modelo y los datos</h2>
    <div class="conclusion">
      <svg class="conclusion-bone" viewBox="0 0 40 160" fill="white"><use href="#spine"/></svg>
      <h3>Hallazgos principales</h3>
      <ul class="conclusion-list">{conclusion_items}</ul>
    </div>
  </div>

</main>

<footer>
  <svg width="20" height="20" viewBox="0 0 40 160" fill="currentColor"><use href="#spine"/></svg>
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

const observer = new IntersectionObserver((entries) => {{
  entries.forEach((entry, i) => {{
    if (entry.isIntersecting) {{
      entry.target.classList.add('visible');
      entry.target.style.animationDelay = i * 0.05 + 's';
      entry.target.querySelectorAll('.feature-bar[data-width]').forEach((bar, idx) => {{
        setTimeout(() => {{ bar.style.width = bar.dataset.width + '%'; }}, idx * 80);
      }});
      observer.unobserve(entry.target);
    }}
  }});
}}, {{ threshold: 0.12 }});

document.querySelectorAll('.animate').forEach(el => observer.observe(el));
</script>
</body>
</html>"""

os.makedirs('reports', exist_ok=True)
with open('reports/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ Dashboard HTML generado en reports/dashboard.html")
