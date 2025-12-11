import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer # Remplace l'ancien 'Imputer'

# Désactiver les avertissements pour un affichage plus propre (spécifiquement les avertissements NumPy)
warnings.filterwarnings("ignore")

# --- PARAMÈTRES DU PROJET ---
DATA_FILE = 'credit_scoring.csv'
# Indices des colonnes supposées:
# Target (y): dernière colonne (13)
# Numériques/Continues (X_num): 0, 2, 3, 7, 8, 9, 10, 11, 12 (9 colonnes)
# Catégorielles (X_cat): 1, 4, 5, 6 (4 colonnes) - Total = 13 (0 à 12)

# La base a 14 colonnes (0 à 13). 
# X = Colonnes 0 à 12 (13 variables)
# Y = Colonne 13 (1 variable)
CONTINUOUS_COLS_INDICES = [0, 2, 3, 7, 8, 9, 10, 11, 12]
CATEGORICAL_COLS_INDICES = [1, 4, 5, 6] # Indices manquants: 1, 4, 5, 6. Total = 4 colonnes

# --- FONCTION run_classifiers SIMPLIFIÉE ---
def run_classifiers(X, y, description, metrics=['accuracy', 'roc_auc']):
    """Exécute des classifieurs, les évalue et affiche les résultats."""
    print(f"\n--- EXÉCUTION DES CLASSIFIEURS: {description} ---")
    
    # Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    classifiers = {
        "Régression Logistique": LogisticRegression(random_state=42, solver='liblinear'),
        "K-Nearest Neighbors (K=5)": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Pour l'AUC
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\nModèle: {name}")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - AUC: {auc:.4f}")
        results[name] = {'Accuracy': acc, 'AUC': auc}
        
    print("-" * 60)
    return results

# ==============================================================================
# SCÉNARIO 1 : VARIABLES CONTINUES SANS DONNÉES MANQUANTES
# ==============================================================================

def scenario_1_no_scaling(df):
    """
    Scénario 1.1: Nettoyage et exécution sur données continues brutes.
    """
    print("\n\n######################################################################")
    print("# SCÉNARIO 1.1: VARIABLES CONTINUES BRUTES (Sans NaN) - BASELINE")
    print("######################################################################")

    # 1. Sélection des variables continues
    X_numeric_df = df.iloc[:, CONTINUOUS_COLS_INDICES].astype(float)
    y_df = df.iloc[:, -1]
    
    # 2. Conversion en NumPy et Binarisation de la cible
    X_numeric = X_numeric_df.values
    
    # La cible est chargée comme string ('+' et '-') ou int (0 et 1), vérifions les valeurs uniques.
    # Pour garantir la binarisation (+ en 1, - en 0)
    y_raw = y_df.values
    if isinstance(y_raw[0], str):
        binarizer = LabelBinarizer(pos_label='+')
        y_bin = binarizer.fit_transform(y_raw).flatten()
    else:
        y_bin = y_raw.astype(int)
    
    # 3. Supprimer les individus contenant des nan (Listwise Deletion)
    rows_with_nan = np.isnan(X_numeric).any(axis=1)
    
    X_clean = X_numeric[~rows_with_nan]
    y_clean = y_bin[~rows_with_nan]

    # 4. Analyse des propriétés
    print(f"Taille de l'échantillon final (lignes, colonnes) : {X_clean.shape}")
    print(f"Nombre d'exemples positifs (1) : {np.sum(y_clean == 1)}")
    print(f"Nombre d'exemples négatifs (0) : {np.sum(y_clean == 0)}")
    
    # 5. Exécution des classifieurs
    results = run_classifiers(
        X_clean, y_clean,
        "Variables Continues BRUTES (Baseline)"
    )
    return results

def scenario_1_with_scaling(df):
    """
    Scénario 1.2: Normalisation des données continues nettoyées (StandardScaler et MinMaxScaler).
    """
    print("\n\n######################################################################")
    print("# SCÉNARIO 1.2: VARIABLES CONTINUES NORMALISÉES")
    print("######################################################################")
    
    # Reprendre les données nettoyées du S1.1
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
    
    # --- Normalisation (StandardScaler) ---
    scaler_std = StandardScaler()
    X_scaled_std = scaler_std.fit_transform(X_clean)
    results_std = run_classifiers(
        X_scaled_std, y_clean,
        "StandardScaler (Centrées-Réduites)"
    )
    
    # --- Normalisation (MinMaxScaler) ---
    scaler_minmax = MinMaxScaler()
    X_scaled_minmax = scaler_minmax.fit_transform(X_clean)
    results_minmax = run_classifiers(
        X_scaled_minmax, y_clean,
        "MinMaxScaler (Entre 0 et 1)"
    )
    return results_std, results_minmax

# ==============================================================================
# SCÉNARIO 2 : TOTALITÉ DE LA BASE (Continues + Catégorielles + Manquantes)
# ==============================================================================

def scenario_2_imputation_encoding_scaling(df):
    """
    Scénario 2: Imputation des manquantes, encodage des catégorielles et normalisation.
    """
    print("\n\n######################################################################")
    print("# SCÉNARIO 2: BASE COMPLÈTE (Imputation + Encodage + Normalisation)")
    print("######################################################################")
    
    # Séparation des X (colonnes 0 à 12) et y (colonne 13)
    X = df.iloc[:, :-1].values
    y_df = df.iloc[:, -1]
    
    # Binarisation de la target (comme précédemment)
    y_raw = y_df.values
    if isinstance(y_raw[0], str):
        binarizer = LabelBinarizer(pos_label='+')
        y_bin = binarizer.fit_transform(y_raw).flatten()
    else:
        y_bin = y_raw.astype(int)
    
    # --- A. Traitement des variables catégorielles (Imputation + OneHotEncoder) ---
    
    # 1. Sélection des catégorielles et conversion en numérique pour Imputer
    X_cat = np.copy(X[:, CATEGORICAL_COLS_INDICES])
    
    for col_id in range(X_cat.shape[1]):
        # Remplacement des valeurs uniques par leur indice numérique (1, 2, 3...)
        # Cela transforme la colonne catégorielle en numérique (ex: ['a', 'b', 'a'] -> [1, 2, 1])
        unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
        X_cat[:, col_id] = val_idx
        
    X_cat = X_cat.astype(float)
    
    # 2. Imputation 'most_frequent' pour les catégorielles (maintenant sous forme numérique)
    # L'indice 0 est la valeur créée pour les NaN lors de la conversion unique (par 'return_inverse').
    # Nous utilisons donc np.nan ici, en supposant que les '?' ont été convertis par pandas.
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent') 
    X_cat_imputed = imp_cat.fit_transform(X_cat)
    
    # 3. Traitement OneHotEncoder
    # X_cat_imputed doit être au format (N_samples, N_features)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_bin = ohe.fit_transform(X_cat_imputed)

    # --- B. Traitement des variables numériques (Imputation 'mean' + Normalisation) ---
    
    # 1. Sélection des numériques et conversion en float
    X_num = np.copy(X[:, CONTINUOUS_COLS_INDICES])
    X_num = X_num.astype(float) # Les NaN sont déjà là si la lecture pandas a bien marché

    # 2. Imputation 'mean' pour les numériques
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_num_imputed = imp_num.fit_transform(X_num)
    
    # 3. Normalisation (StandardScaler) pour les numériques imputées
    scaler_std = StandardScaler()
    X_num_scaled = scaler_std.fit_transform(X_num_imputed)

    # --- C. Construction du jeu de données final ---
    # Concaténer les numériques normalisées et les catégorielles binarisées
    X_final = np.concatenate((X_num_scaled, X_cat_bin), axis=1)
    
    print(f"Taille de l'échantillon final complet (lignes, colonnes) : {X_final.shape}")
    print(f"Nombre total de caractéristiques après encodage OHE : {X_final.shape[1]}")

    # 4. Exécution des classifieurs
    results = run_classifiers(
        X_final, y_bin,
        "Base Complète (Imputation + Encodage + Normalisation)"
    )
    return results


def main():
    """Fonction principale d'exécution du projet."""
    
    print("--- DÉMARRAGE DE L'ANALYSE ---")
    
    # Chargement initial des données avec traitement des '?' en NaN
    try:
        df = pd.read_csv(
            DATA_FILE,
            sep=';',
            header=None,
            skiprows=1,
            na_values='?'
        )
    except FileNotFoundError:
        print(f"\n❌ ERREUR: Le fichier '{DATA_FILE}' est introuvable. Assurez-vous qu'il est dans le répertoire.")
        return
        
    # --- 1. Exécution des scénarios ---
    
    # Scénario 1.1: Continues brutes (Baseline)
    results_s1_brut = scenario_1_no_scaling(df.copy())
    
    # Scénario 1.2: Continues normalisées
    results_s1_std, results_s1_minmax = scenario_1_with_scaling(df.copy())
    
    # Scénario 2: Base complète imputée, encodée et normalisée
    results_s2_full = scenario_2_imputation_encoding_scaling(df.copy())
    
    
    # --- 2. Interprétation et Synthèse des Résultats ---
    
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
        'LogReg_AUC': [
            results_s1_brut['Régression Logistique']['AUC'],
            results_s1_std['Régression Logistique']['AUC'],
            results_s1_minmax['Régression Logistique']['AUC'],
            results_s2_full['Régression Logistique']['AUC']
        ],
        'KNN_AUC': [
            results_s1_brut['K-Nearest Neighbors (K=5)']['AUC'],
            results_s1_std['K-Nearest Neighbors (K=5)']['AUC'],
            results_s1_minmax['K-Nearest Neighbors (K=5)']['AUC'],
            results_s2_full['K-Nearest Neighbors (K=5)']['AUC']
        ]
    }

    summary_df = pd.DataFrame(data)
    
    print(summary_df.to_markdown(index=False))
    
    print("\n\n--- INTERPRÉTATION FINALE ---")
    print("\nComparaison S1.1 (Brut) vs S1.2 (Normalisé) :")
    print("La Régression Logistique (LogReg) est généralement peu affectée par la normalisation.")
    print("Le K-Nearest Neighbors (KNN), basé sur la distance, DOIT voir son AUC augmenter après la normalisation (Standard/MinMax Scaler), car toutes les variables ont maintenant la même échelle d'importance. Vérifiez si l'AUC de KNN s'améliore entre S1.1 et S1.2.")
    
    print("\nComparaison S1 (Nettoyé) vs S2 (Complet) :")
    print("Le Scénario 2 utilise l'intégralité de la base et toutes les variables (continues ET catégorielles). Si le traitement (imputation, OHE) a été efficace, le modèle le plus performant (LogReg) devrait idéalement montrer une légère amélioration de son AUC en Scénario 2, en exploitant l'information des variables catégorielles et des individus initialement retirés.")


if __name__ == '__main__':
    main()