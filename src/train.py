import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle

# Cargar datos
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

label_names = ['DH', 'NO', 'SL']

# Modelos — dataset pequeño, probamos más variedad
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
}

results = {}
print("=== ENTRENANDO MODELOS ===\n")

for name, model in models.items():
    # Cross-validation — con 310 pacientes es más confiable que un solo split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'model': model, 'f1': f1, 'cv_mean': cv_scores.mean(), 'y_pred': y_pred}

    print(f"--- {name} ---")
    print(f"F1 Test: {f1:.4f} | CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(classification_report(y_test, y_pred, target_names=label_names))

# Mejor modelo
best_name = max(results, key=lambda x: results[x]['f1'])
print(f"\n✅ Mejor modelo: {best_name} (F1: {results[best_name]['f1']:.4f})")

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(results[best_name]['model'], f)

print("Modelo guardado en models/")