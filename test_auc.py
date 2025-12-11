import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

DATA_FILE = 'credit_scoring.csv'
CONTINUOUS_COLS_INDICES = [0, 2, 3, 7, 8, 9, 10, 11, 12]
CATEGORICAL_COLS_INDICES = [1, 4, 5, 6]

def run_classifiers(X, y, description, metrics=['accuracy', 'roc_auc']):
    """
    Exécute les classifieurs spécifiques (CART, KNN, MLP), les évalue et affiche les résultats.
    """
    print(f"\n--- EXÉCUTION DES CLASSIFIEURS: {description} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    classifiers = {
        'CART': DecisionTreeClassifier(random_state=1),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'MLP (40-20)': MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000)
    }

    results = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\nModèle: {name}")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - AUC: {auc:.4f}")
        results[name] = {'Accuracy': acc, 'AUC': auc}
        
    print("-" * 60)
    return results

# Scénario 1 (variables continues ss données manquantes)
def scenario_1_no_scaling(df):
    """Scénario 1.1: Nettoyage et exécution sur données continues brutes."""
    print("\n\n######################################################################")
    print("# SCÉNARIO 1.1: VARIABLES CONTINUES BRUTES (Sans NaN) - BASELINE")
    print("######################################################################")

    X_numeric_df = df.iloc[:, CONTINUOUS_COLS_INDICES].astype(float)
    y_df = df.iloc[:, -1]
    
    X_numeric = X_numeric_df.values
    
    y_raw = y_df.values
    if isinstance(y_raw[0], str):
        binarizer = LabelBinarizer(pos_label='+')
        y_bin = binarizer.fit_transform(y_raw).flatten()
    else:
        y_bin = y_raw.astype(int)
    
    rows_with_nan = np.isnan(X_numeric).any(axis=1)
    
    X_clean = X_numeric[~rows_with_nan]
    y_clean = y_bin[~rows_with_nan]

    print(f"Taille de l'échantillon final (lignes, colonnes) : {X_clean.shape}")
    print(f"Nombre d'exemples positifs (1) : {np.sum(y_clean == 1)}")
    print(f"Nombre d'exemples négatifs (0) : {np.sum(y_clean == 0)}")
    
    results = run_classifiers(
        X_clean, y_clean,
        "Variables Continues BRUTES (Baseline)"
    )
    return results

def scenario_1_with_scaling(df):
    """Scénario 1.2: Normalisation des données continues nettoyées (StandardScaler et MinMaxScaler)."""
    print("\n\n######################################################################")
    print("# SCÉNARIO 1.2: VARIABLES CONTINUES NORMALISÉES")
    print("######################################################################")
    
    X_numeric_df = df.iloc[:, CONTINUOUS_COLS_INDICES].astype(float)
    y_df = df.iloc[:, -1]
    
    y_raw = y_df.values
    if isinstance(y_raw[0], str):
        binarizer = LabelBinarizer(pos_label='+')
        y_bin = binarizer.fit_transform(y_raw).flatten()
    else:
        y_bin = y_raw.astype(int)
    
    X_numeric = X_numeric_df.values
    rows_with_nan = np.isnan(X_numeric).any(axis=1)
    X_clean = X_numeric[~rows_with_nan]
    y_clean = y_bin[~rows_with_nan]
    
    scaler_std = StandardScaler()
    X_scaled_std = scaler_std.fit_transform(X_clean)
    results_std = run_classifiers(
        X_scaled_std, y_clean,
        "StandardScaler (Centrées-Réduites)"
    )
    
    scaler_minmax = MinMaxScaler()
    X_scaled_minmax = scaler_minmax.fit_transform(X_clean)
    results_minmax = run_classifiers(
        X_scaled_minmax, y_clean,
        "MinMaxScaler (Entre 0 et 1)"
    )
    return results_std, results_minmax

# Scénario 2 (totalité de la bse)
def scenario_2_imputation_encoding_scaling(df):
    """Scénario 2: Imputation des manquantes, encodage des catégorielles et normalisation."""
    print("\n\n######################################################################")
    print("# SCÉNARIO 2: BASE COMPLÈTE (Imputation + Encodage + Normalisation)")
    print("######################################################################")
    
    X = df.iloc[:, :-1].values
    y_df = df.iloc[:, -1]

    y_raw = y_df.values
    if isinstance(y_raw[0], str):
        binarizer = LabelBinarizer(pos_label='+')
        y_bin = binarizer.fit_transform(y_raw).flatten()
    else:
        y_bin = y_raw.astype(int)
    
    
    X_cat = np.copy(X[:, CATEGORICAL_COLS_INDICES])
    
    for col_id in range(X_cat.shape[1]):
        unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
        X_cat[:, col_id] = val_idx
        
    X_cat = X_cat.astype(float)
    
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent') 
    X_cat_imputed = imp_cat.fit_transform(X_cat)
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_bin = ohe.fit_transform(X_cat_imputed)


    X_num = np.copy(X[:, CONTINUOUS_COLS_INDICES])
    X_num = X_num.astype(float) 

    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_num_imputed = imp_num.fit_transform(X_num)

    scaler_std = StandardScaler()
    X_num_scaled = scaler_std.fit_transform(X_num_imputed)

    X_final = np.concatenate((X_num_scaled, X_cat_bin), axis=1)
    
    print(f"Taille de l'échantillon final complet (lignes, colonnes) : {X_final.shape}")
    print(f"Nombre total de caractéristiques après encodage OHE : {X_final.shape[1]}")

    results = run_classifiers(
        X_final, y_bin,
        "Base Complète (Imputation + Encodage + Normalisation)"
    )
    return results


def main():
    """Fonction principale d'exécution du projet."""
    
    print("--- DÉMARRAGE DE L'ANALYSE ---")
    
    try:
        df = pd.read_csv(
            DATA_FILE,
            sep=';',
            header=None,
            skiprows=1,
            na_values='?'
        )
    except FileNotFoundError:
        print(f"\nERREUR: Le fichier '{DATA_FILE}' est introuvable. Assurez-vous qu'il est dans le répertoire.")
        return
    
    results_s1_brut = scenario_1_no_scaling(df.copy())
    results_s1_std, results_s1_minmax = scenario_1_with_scaling(df.copy())
    results_s2_full = scenario_2_imputation_encoding_scaling(df.copy())
    

    print("\n\n\n======================================================================")
    print("                      SYNTHÈSE DES RÉSULTATS (AUC)                     ")
    print("======================================================================")

    data = {
        'Scenario': [
            '1.1 Continues BRUTES (Baseline)',
            '1.2 Continues STANDARD SCALER',
            '1.2 Continues MINMAX SCALER',
            '2. Base COMPLÈTE (Imputation + Encodage + Standard Scaler)'
        ],
        'CART_AUC': [
            results_s1_brut['CART']['AUC'],
            results_s1_std['CART']['AUC'],
            results_s1_minmax['CART']['AUC'],
            results_s2_full['CART']['AUC']
        ],
        'KNN_AUC': [
            results_s1_brut['KNN (k=5)']['AUC'],
            results_s1_std['KNN (k=5)']['AUC'],
            results_s1_minmax['KNN (k=5)']['AUC'],
            results_s2_full['KNN (k=5)']['AUC']
        ],
        'MLP_AUC': [
            results_s1_brut['MLP (40-20)']['AUC'],
            results_s1_std['MLP (40-20)']['AUC'],
            results_s1_minmax['MLP (40-20)']['AUC'],
            results_s2_full['MLP (40-20)']['AUC']
        ]
    }

    summary_df = pd.DataFrame(data)
    
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    
    print("\n\n--- INTERPRÉTATION FINALE ---")
    print("1. CART (Arbre de Décision) : Est peu sensible à la normalisation (comparer S1.1 et S1.2).")
    print("2. KNN et MLP : Sont très sensibles à la mise à l'échelle (normalisation). Vous devriez voir leur AUC augmenter entre S1.1 et S1.2.")
    print("3. Scénario 2 : Le Scénario 2, utilisant l'ensemble des variables (continues et catégorielles), devrait fournir la meilleure AUC globale pour le modèle le plus performant.")


if __name__ == '__main__':
    main()