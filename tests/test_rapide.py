"""
Test rapide de train.py — sans MLflow, sans infrastructure.

Lance directement :
    cd mlops_drift
    python tests/test_rapide.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.generator import (
    compute_feature_stats,
    generate_drifted_data,
    generate_reference_data,
)
from src.training.train import train, evaluate_model


def sep(titre=""):
    print(f"\n{'─'*55}")
    if titre:
        print(f"  {titre}")
        print(f"{'─'*55}")


# ─────────────────────────────────────────────────────
# 1. Génération des données
# ─────────────────────────────────────────────────────
sep("1. GÉNÉRATION DES DONNÉES")

df_ref   = generate_reference_data(n_rows=3000, seed=42)
df_drift = generate_drifted_data(n_rows=3000, seed=42)

print(f"\n  Référence  : {df_ref.shape[0]} lignes × {df_ref.shape[1]} colonnes")
print(f"  Driftée    : {df_drift.shape[0]} lignes × {df_drift.shape[1]} colonnes")

print(f"\n  {'Feature':<8} {'Réf mean':>10} {'Drift mean':>12}  {'⚠️ drift ?':>10}")
print(f"  {'-'*47}")
for feat in ["x1", "x2", "x3", "x4"]:
    r = df_ref[feat].mean()
    d = df_drift[feat].mean()
    flag = "⚠️  OUI" if abs(d - r) > 0.3 else "✅ non"
    print(f"  {feat:<8} {r:>10.3f} {d:>12.3f}  {flag}")

# ─────────────────────────────────────────────────────
# 2. Entraînement sur données de référence
# ─────────────────────────────────────────────────────
sep("2. ENTRAÎNEMENT — DONNÉES RÉFÉRENCE")

model_rf, metrics_rf, X_val, y_val = train(
    df_ref,
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": 6,
                  "min_samples_leaf": 4, "random_state": 42, "n_jobs": -1},
)

print(f"\n  Modèle     : RandomForest (100 arbres, depth=6)")
print(f"  Train RMSE : {metrics_rf['train_rmse']:.4f}")
print(f"  Val   RMSE : {metrics_rf['val_rmse']:.4f}")
print(f"  Val   MAE  : {metrics_rf['val_mae']:.4f}")
print(f"  Val   R²   : {metrics_rf['val_r2']:.4f}  {'✅ bon' if metrics_rf['val_r2'] > 0.8 else '⚠️ à surveiller'}")
print(f"  Durée      : {metrics_rf['train_duration_sec']:.2f}s")
print(f"  Lignes     : {metrics_rf['n_rows_train']} train / {metrics_rf['n_rows_val']} val")

# ─────────────────────────────────────────────────────
# 3. Baseline linéaire pour comparaison
# ─────────────────────────────────────────────────────
sep("3. BASELINE — RIDGE (modèle linéaire)")

model_ridge, metrics_ridge, _, _ = train(
    df_ref,
    model_type="ridge",
    model_params={"alpha": 1.0},
)

print(f"\n  Modèle     : Ridge (alpha=1.0)")
print(f"  Val   RMSE : {metrics_ridge['val_rmse']:.4f}")
print(f"  Val   R²   : {metrics_ridge['val_r2']:.4f}")
print(f"\n  💡 Ridge ≈ RF car la relation y=f(x) est linéaire par construction.")
print(f"     En données réelles, le RF surpasse généralement Ridge.")

# ─────────────────────────────────────────────────────
# 4. Entraînement sur données driftées
# ─────────────────────────────────────────────────────
sep("4. ENTRAÎNEMENT — DONNÉES DRIFTÉES")

model_drifted, metrics_drifted, _, _ = train(
    df_drift,
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": 6,
                  "min_samples_leaf": 4, "random_state": 42, "n_jobs": -1},
)

print(f"\n  Val R² sur données driftées  : {metrics_drifted['val_r2']:.4f}")
print(f"\n  💡 Le R² est bon car le modèle est entraîné ET évalué sur la")
print(f"     même distribution driftée. Le problème réel, c'est d'appliquer")
print(f"     le modèle 'référence' sur des données driftées (étape 5).")

# ─────────────────────────────────────────────────────
# 5. Simulation du vrai problème de drift
# ─────────────────────────────────────────────────────
sep("5. SIMULATION DU PROBLÈME DE DRIFT")
print("  On prend le modèle entraîné sur la RÉFÉRENCE")
print("  et on l'évalue sur les données DRIFTÉES.")
print("  C'est exactement ce qui se passe en production.\n")

features = [c for c in df_drift.columns if c != "y"]
metrics_cross = evaluate_model(model_rf, df_drift[features], df_drift["y"])

print(f"  Modèle RF (entraîné sur référence)")
print(f"  {'':4} {'Sur référence':>18}  {'Sur données driftées':>22}  {'Dégradation':>12}")
print(f"  {'-'*65}")

for metric in ["rmse", "mae", "r2"]:
    val_ref   = metrics_rf[f"val_{metric}"]
    val_drift = metrics_cross[metric]
    if metric == "r2":
        delta = val_drift - val_ref
        flag  = "⚠️  dégradé" if delta < -0.05 else "✅ stable"
    else:
        delta = val_drift - val_ref
        flag  = "⚠️  dégradé" if delta > 0.05 else "✅ stable"
    print(f"  {metric.upper():<6} {val_ref:>18.4f}  {val_drift:>22.4f}  {delta:>+10.4f}  {flag}")

print(f"\n  💡 C'est cette dégradation que le pipeline de monitoring doit")
print(f"     détecter AVANT qu'elle n'impacte les prédictions en production.")

# ─────────────────────────────────────────────────────
# 6. Stats de référence (ce qui serait stocké dans Postgres)
# ─────────────────────────────────────────────────────
sep("6. STATS DE RÉFÉRENCE (aperçu de ce qui est stocké)")
print("  Ces stats sont sauvegardées dans Postgres après chaque entraînement.")
print("  Le pipeline de monitoring les compare aux nouvelles données.\n")

stats = compute_feature_stats(df_ref)
print(stats.round(3).to_string())

# ─────────────────────────────────────────────────────
# 7. Feature importance
# ─────────────────────────────────────────────────────
sep("7. FEATURE IMPORTANCE (RandomForest)")

importances = list(zip(features, model_rf.feature_importances_))
importances_sorted = sorted(importances, key=lambda x: x[1], reverse=True)
features_sorted = [feat for feat, imp in importances_sorted]

print(importances)

for feat, imp in sorted(importances, key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 60)
    print(f"  {feat}  {bar}  {imp:.4f}")

print(f"\n  💡 {features_sorted[0]} et {features_sorted[1]}")
print(f"     sont les plus grands dans la formule de y.")

# ─────────────────────────────────────────────────────
# Bilan final
# ─────────────────────────────────────────────────────
sep("BILAN")
print(f"""
  ✅  Génération de données          OK
  ✅  Entraînement RandomForest      OK  (R²={metrics_rf['val_r2']:.4f})
  ✅  Entraînement Ridge (baseline)  OK  (R²={metrics_ridge['val_r2']:.4f})
  ✅  Entraînement sur drift         OK
  ✅  Simulation dégradation prod.   OK
  ✅  Calcul stats de référence      OK
  ✅  Feature importance             OK

  train.py fonctionne correctement.
  Prochaine étape : les DAGs Airflow qui orchestrent tout ça.
""")