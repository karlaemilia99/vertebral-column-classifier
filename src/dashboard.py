import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pickle

# Cargar datos
cols = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class'
]
df_orig = pd.read_csv('data/column_3C.dat', sep=' ', names=cols)

X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

label_names = ['DH', 'NO', 'SL']
COLORS_CLASS = {'DH': '#e74c3c', 'NO': '#2ecc71', 'SL': '#3498db'}
COLORS = ['#00d4ff', '#51cf66', '#ff6b6b', '#fcc419']
TEXT  = '#e0e0e0'
GRID  = '#2a2a3e'

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'model': model, 'f1': f1, 'cv': cv.mean(), 'y_pred': y_pred}

best_name = max(results, key=lambda x: results[x]['f1'])

def style_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle='--', linewidth=0.5)

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. PCA — visualización 2D
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all)
class_colors = [list(COLORS_CLASS.values())[int(c)] for c in y_all]
for i, (cls, color) in enumerate(COLORS_CLASS.items()):
    mask = y_all == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, alpha=0.6, s=30, label=cls, edgecolors='none')
ax1.set_title('Distribución PCA — 3 Clases', fontweight='bold')
ax1.set_xlabel('Componente 1')
ax1.set_ylabel('Componente 2')
ax1.legend(fontsize=9, labelcolor=TEXT, facecolor='#1a1a2e', edgecolor=GRID)

# 2. Comparación F1 Test vs CV
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
names = list(results.keys())
f1s  = [results[n]['f1'] for n in names]
cvs  = [results[n]['cv'] for n in names]
x = np.arange(len(names))
w = 0.35
ax2.bar(x - w/2, f1s, w, color='#00d4ff', label='F1 Test', edgecolor='none')
ax2.bar(x + w/2, cvs, w, color='#ff6b6b', label='CV Mean', edgecolor='none')
ax2.set_xticks(x)
ax2.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
ax2.set_ylim(0.6, 1.0)
ax2.set_title('F1 Test vs Cross-Validation', fontweight='bold')
ax2.legend(fontsize=8, labelcolor=TEXT, facecolor='#1a1a2e', edgecolor=GRID)

# 3. Matriz de confusión
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
im = ax3.imshow(cm, cmap='Blues', aspect='auto')
ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
ax3.set_xticklabels(label_names, color=TEXT)
ax3.set_yticklabels(label_names, color=TEXT)
for i in range(3):
    for j in range(3):
        ax3.text(j, i, str(cm[i,j]), ha='center', va='center',
                 color='white' if cm[i,j] > cm.max()/2 else TEXT, fontsize=12)
ax3.set_title(f'Matriz de Confusión — {best_name}', fontweight='bold')
ax3.set_xlabel('Predicho')
ax3.set_ylabel('Real')

# 4. Feature importance
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4)
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()
ax4.barh(importances.index, importances.values, color='#00d4ff', edgecolor='none', height=0.6)
ax4.set_title('Feature Importance — Random Forest', fontweight='bold')
ax4.set_xlabel('Importancia')

# 5. Boxplot degree_spondylolisthesis por clase
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)
for i, (cls, color) in enumerate(COLORS_CLASS.items()):
    data = df_orig[df_orig['class'] == cls]['degree_spondylolisthesis']
    bp = ax5.boxplot(data, positions=[i], widths=0.5,
                     patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.6),
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(color=TEXT),
                     capprops=dict(color=TEXT),
                     flierprops=dict(markerfacecolor=color, markersize=4))
ax5.set_xticks([0,1,2])
ax5.set_xticklabels(list(COLORS_CLASS.keys()))
ax5.set_title('Degree Spondylolisthesis por Clase', fontweight='bold')
ax5.set_ylabel('Grado')

# 6. F1 por clase — mejor modelo
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6)
from sklearn.metrics import classification_report
report = classification_report(y_test, results[best_name]['y_pred'],
                                target_names=label_names, output_dict=True)
classes = label_names
f1_per_class = [report[c]['f1-score'] for c in classes]
bars = ax6.bar(classes, f1_per_class,
               color=list(COLORS_CLASS.values()), edgecolor='none', width=0.5)
ax6.set_ylim(0, 1.1)
ax6.set_title(f'F1-score por Clase — {best_name}', fontweight='bold')
ax6.set_ylabel('F1-score')
for bar, val in zip(bars, f1_per_class):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.2f}', ha='center', color=TEXT, fontsize=10)

fig.text(0.5, 0.97, 'Vertebral Column — Clasificación de Patologías Ortopédicas',
         ha='center', fontsize=16, fontweight='bold', color=TEXT)
fig.text(0.5, 0.94, 'Random Forest · Logistic Regression · SVM · KNN  |  310 pacientes  |  6 variables biomecánicas',
         ha='center', fontsize=9, color='#888888')

plt.savefig('reports/dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Dashboard guardado en reports/dashboard.png")