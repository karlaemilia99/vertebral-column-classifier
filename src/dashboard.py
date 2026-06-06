import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import pickle
import os

# Cargar datos
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

best_name = max(results, key=lambda x: results[x]['f1'])
best_pred = results[best_name]['y_pred']
best_f1   = results[best_name]['f1']

# Métricas por clase
report = classification_report(y_test, best_pred, target_names=label_names, output_dict=True)
cm = confusion_matrix(y_test, best_pred)

# Feature importance
rf = results['Random Forest']['model']
cols = X_train.columns.tolist()
importances = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)

# Datos para el HTML
models_f1  = {n: round(results[n]['f1'], 4) for n in results}
models_cv  = {n: round(results[n]['cv'], 4) for n in results}
f1_per_cls = {cls: round(report[cls]['f1-score'], 2) for cls in label_names}
cm_vals    = cm.tolist()
feat_data  = [(col, round(val, 4)) for col, val in importances.items()]
max_feat   = feat_data[0][1]

print(f"\n✅ Mejor modelo: {best_name} (F1: {best_f1:.4f})")

# ── GENERAR HTML ───────────────────────────────────────────
models_list   = list(models_f1.keys())
f1_list       = list(models_f1.values())
cv_list       = list(models_cv.values())
class_f1_list = list(f1_per_cls.values())

feature_bars_html = ''
for name, val in feat_data:
    pct = round((val / max_feat) * 100, 1)
    feature_bars_html += f'''
    <div class="feature-row">
      <span class="feature-name">{name.replace("_", " ").title()}</span>
      <div class="feature-bar-wrap">
        <div class="feature-bar" style="width:{pct}%"></div>
      </div>
      <span class="feature-val">{round(val*100, 1)}%</span>
    </div>'''

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vertebral Column — Reporte de Resultados</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #f7f4ef; --bg2: #efeae2; --white: #ffffff;
    --ink: #1a1612; --ink2: #4a4540; --ink3: #8a837a;
    --dh: #c0392b; --dh-light: #fadbd8;
    --no: #1a7a4a; --no-light: #d5f0e3;
    --sl: #1a5fa8; --sl-light: #d0e4f7;
    --accent: #b5763a; --border: #ddd6ca;
  }}
  html {{ scroll-behavior: smooth; }}
  body {{ font-family: 'Source Sans 3', sans-serif; background: var(--bg); color: var(--ink); line-height: 1.7; font-size: 16px; }}
  header {{ background: var(--ink); color: var(--bg); padding: 5rem 3rem 4rem; position: relative; overflow: hidden; }}
  header::after {{ content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--dh), var(--no), var(--sl)); }}
  .header-inner {{ max-width: 900px; margin: 0 auto; }}
  .header-tag {{ font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--accent); font-weight: 500; margin-bottom: 1.5rem; display: block; }}
  header h1 {{ font-family: 'Playfair Display', serif; font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 400; line-height: 1.15; margin-bottom: 1.5rem; }}
  header h1 em {{ font-style: italic; color: var(--accent); }}
  .header-meta {{ display: flex; gap: 2.5rem; flex-wrap: wrap; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1); }}
  .meta-item {{ display: flex; flex-direction: column; gap: 0.2rem; }}
  .meta-val {{ font-family: 'Playfair Display', serif; font-size: 1.8rem; color: var(--accent); line-height: 1; }}
  .meta-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(247,244,239,0.5); }}
  main {{ max-width: 1000px; margin: 0 auto; padding: 4rem 2rem; }}
  .section {{ margin-bottom: 5rem; }}
  .section-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.2em; color: var(--accent); font-weight: 500; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.75rem; }}
  .section-label::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
  .section-title {{ font-family: 'Playfair Display', serif; font-size: clamp(1.5rem, 3vw, 2.2rem); font-weight: 400; margin-bottom: 1rem; line-height: 1.2; }}
  .section-intro {{ font-size: 1rem; color: var(--ink2); max-width: 680px; margin-bottom: 2.5rem; font-weight: 300; line-height: 1.8; }}
  .diagnosis-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 3rem; }}
  .dx-card {{ background: var(--white); border: 1px solid var(--border); border-radius: 4px; padding: 1.75rem; position: relative; overflow: hidden; transition: transform 0.2s, box-shadow 0.2s; }}
  .dx-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
  .dx-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; }}
  .dx-card.dh::before {{ background: var(--dh); }}
  .dx-card.no::before {{ background: var(--no); }}
  .dx-card.sl::before {{ background: var(--sl); }}
  .dx-badge {{ display: inline-block; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.3rem 0.75rem; border-radius: 2px; margin-bottom: 1rem; }}
  .dh .dx-badge {{ background: var(--dh-light); color: var(--dh); }}
  .no .dx-badge {{ background: var(--no-light); color: var(--no); }}
  .sl .dx-badge {{ background: var(--sl-light); color: var(--sl); }}
  .dx-name {{ font-family: 'Playfair Display', serif; font-size: 1.2rem; margin-bottom: 0.5rem; }}
  .dx-desc {{ font-size: 0.875rem; color: var(--ink2); font-weight: 300; line-height: 1.7; margin-bottom: 1.25rem; }}
  .dx-stats {{ display: flex; gap: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
  .dx-stat-num {{ font-family: 'Playfair Display', serif; font-size: 1.5rem; display: block; line-height: 1; margin-bottom: 0.2rem; }}
  .dh .dx-stat-num {{ color: var(--dh); }} .no .dx-stat-num {{ color: var(--no); }} .sl .dx-stat-num {{ color: var(--sl); }}
  .dx-stat-label {{ font-size: 0.72rem; color: var(--ink3); text-transform: uppercase; letter-spacing: 0.06em; }}
  .results-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-card {{ background: var(--white); border: 1px solid var(--border); border-radius: 4px; padding: 1.75rem; }}
  .chart-title {{ font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; }}
  .chart-subtitle {{ font-size: 0.825rem; color: var(--ink3); margin-bottom: 1.25rem; font-weight: 300; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .insights-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 2rem; }}
  .insight {{ background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 0 4px 4px 0; padding: 1.25rem 1.5rem; }}
  .insight-title {{ font-weight: 500; font-size: 0.875rem; margin-bottom: 0.5rem; }}
  .insight-text {{ font-size: 0.85rem; color: var(--ink2); font-weight: 300; line-height: 1.7; }}
  .cm-wrap {{ background: var(--white); border: 1px solid var(--border); border-radius: 4px; padding: 1.75rem; margin-bottom: 1.5rem; }}
  .cm-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-top: 1rem; }}
  .cm-table th {{ padding: 0.6rem 1rem; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink3); text-align: center; }}
  .cm-table td {{ padding: 0.75rem 1rem; text-align: center; border: 1px solid var(--border); font-family: 'Playfair Display', serif; font-size: 1.2rem; }}
  .cm-label {{ font-family: 'Source Sans 3', sans-serif !important; font-size: 0.8rem !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink2); }}
  .cm-correct {{ background: rgba(26,122,74,0.08); color: var(--no); }}
  .cm-wrong   {{ background: rgba(192,57,43,0.06); color: var(--dh); }}
  .feature-list {{ display: flex; flex-direction: column; gap: 0.75rem; margin-top: 1rem; }}
  .feature-row {{ display: flex; align-items: center; gap: 1rem; }}
  .feature-name {{ font-size: 0.825rem; color: var(--ink2); width: 220px; flex-shrink: 0; font-weight: 300; }}
  .feature-bar-wrap {{ flex: 1; background: var(--bg2); border-radius: 2px; height: 8px; overflow: hidden; }}
  .feature-bar {{ height: 100%; border-radius: 2px; background: var(--accent); }}
  .feature-val {{ font-size: 0.8rem; color: var(--ink3); width: 40px; text-align: right; }}
  .conclusion {{ background: var(--ink); color: var(--bg); border-radius: 4px; padding: 3rem; margin-top: 2rem; }}
  .conclusion h3 {{ font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 400; margin-bottom: 1.5rem; color: var(--accent); }}
  .conclusion-list {{ list-style: none; display: flex; flex-direction: column; gap: 1rem; }}
  .conclusion-list li {{ display: flex; gap: 1rem; font-size: 0.925rem; color: rgba(247,244,239,0.8); font-weight: 300; line-height: 1.7; }}
  .conclusion-list li::before {{ content: '→'; color: var(--accent); flex-shrink: 0; }}
  footer {{ text-align: center; padding: 2rem; font-size: 0.8rem; color: var(--ink3); border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <span class="header-tag">Reporte de Resultados — Machine Learning</span>
    <h1>Clasificación de<br>Patologías de <em>Columna Vertebral</em></h1>
    <div class="header-meta">
      <div class="meta-item"><span class="meta-val">310</span><span class="meta-label">Pacientes analizados</span></div>
      <div class="meta-item"><span class="meta-val"
