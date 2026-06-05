import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Cargar datos
cols = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class'
]
df = pd.read_csv('data/column_3C.dat', sep=' ', names=cols)

# Separar X e y
X = df.drop(columns=['class'])
y = df['class']

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("=== CLASES CODIFICADAS ===")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# StandardScaler — dataset pequeño y sin outliers extremos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n=== SPLIT ===")
print(f"Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")
print(f"\nDistribución train:")
unique, counts = np.unique(y_train, return_counts=True)
for cls_idx, count in zip(unique, counts):
    print(f"  {le.classes_[cls_idx]}: {count}")

# Guardar
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/X_test.csv', index=False)
pd.Series(y_train).to_csv('data/y_train.csv', index=False)
pd.Series(y_test).to_csv('data/y_test.csv', index=False)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nArchivos guardados.")