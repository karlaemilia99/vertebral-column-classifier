import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# El archivo no tiene header — lo definimos manualmente
cols = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class'
]

df = pd.read_csv('data/column_3C.dat', sep=' ', names=cols)

print("=== SHAPE ===")
print(df.shape)

print("\n=== PRIMERAS FILAS ===")
print(df.head())

print("\n=== DISTRIBUCIÓN DE CLASES ===")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True).round(2))

print("\n=== VALORES NULOS ===")
print(df.isnull().sum().sum())

print("\n=== ESTADÍSTICAS ===")
print(df.describe().round(2))

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Vertebral Column — Exploración por Clase', fontsize=16, fontweight='bold')

features = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'
]

colors = {'DH': '#e74c3c', 'NO': '#2ecc71', 'SL': '#3498db'}

for ax, feature in zip(axes.flatten(), features):
    for cls in df['class'].unique():
        subset = df[df['class'] == cls][feature]
        ax.hist(subset, bins=20, alpha=0.6, label=cls,
                color=colors[cls], edgecolor='none')
    ax.set_title(feature.replace('_', ' ').title())
    ax.set_xlabel('Valor')
    ax.set_ylabel('Pacientes')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('reports/eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada en reports/eda.png")