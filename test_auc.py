import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Désactiver les avertissements pour un affichage plus propre
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_data(file_path='credit_scoring.csv'):
    """
    Charge, nettoie et prépare le jeu de données selon les étapes de l'énoncé.
    """
    print("--- 1. PRÉPARATION DES DONNÉES ---")
    
    # 1. Chargement des données (skiprows=1 pour ignorer l'en-tête)
    df = pd.read_csv(
        file_path,
        sep=';',
        header=None,
        skiprows=1,  # Correction: Ignore la ligne d'en-tête
        na_values='?'
    )

    # Séparer les variables caractéristiques (X) de la variable à prédire (y)
    X_df = df.iloc[:, :-1].copy()
    y_df = df.iloc[:, -1].copy()

    # 2. Créer un sous-ensemble de vos données en gardant que les variables numériques (continues)
    # Indices des colonnes continues présumées: 0, 2, 3, 7, 8, 9, 10, 11, 12
    # (Seniority, Time, Age, Expenses, Income, Assets, Debt, Amount, Price)
    continuous_cols_indices = [0, 2, 3, 7, 8, 9, 10, 11, 12]
    
    X_numeric_df = X_df[continuous_cols_indices]
    
    # Transformation en numpy Array et typage en float (les NaN sont inclus)
    X_numeric = X_numeric_df.values.astype(float)
    y_all = y_df.values.astype(int) # La cible est déjà en 0/1

    # 3. Supprimer les individus dans vos données contenant des nan sur au moins une variable.
    rows_with_nan = np.isnan(X_numeric).any(axis=1)
    
    X_clean = X_numeric[~rows_with_nan]
    y_clean = y_all[~rows_with_nan]

    # 4. Binarisation (vérification): La cible est déjà en 0/1 (int), nous la laissons telle quelle
    y_bin = y_clean
    
    # 5. Analyse des propriétés (pour l'énoncé)
    print(f"Taille de l'échantillon final (lignes, colonnes) : {X_clean.shape}")
    print(f"Nombre d'exemples positifs (1) : {np.sum(y_bin == 1)}")
    print(f"Nombre d'exemples négatifs (0) : {np.sum(y_bin == 0)}")
    print("-" * 50)

    return X_clean, y_bin


def run_classifiers(X, y, metrics):
    """
    Exécute des classifieurs, les évalue en utilisant l'Accuracy et l'AUC.
    """
    print("--- 2. EXÉCUTION DES CLASSIFIEURS ---")
    
    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    classifiers = {
        "Régression Logistique": LogisticRegression(random_state=42, solver='liblinear'),
        "K-Nearest Neighbors (K=5)": KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Pour l'AUC, nous utilisons predict_proba
        y_proba = model.predict_proba(X_test)[:, 1]
            
        print(f"\nModèle: {name}")
        
        if 'accuracy' in metrics:
            acc = accuracy_score(y_test, y_pred)
            print(f"  - Accuracy: {acc:.4f}")
            
        if 'roc_auc' in metrics:
            auc = roc_auc_score(y_test, y_proba)
            print(f"  - AUC (Aire sous la courbe ROC): {auc:.4f}")
            
    print("-" * 50)


if __name__ == '__main__':
    data_file = 'credit_scoring.csv'
    
    try:
        # Étape 1: Préparation des données
        X, y = prepare_data(data_file)
        
        # Étape 2: Exécution des classifieurs avec l'AUC
        run_classifiers(X, y, metrics=['accuracy', 'roc_auc'])
        
        print("\n✅ EXÉCUTION TERMINÉE.")
        print("Interprétation : Comparez les valeurs d'AUC. Plus elles sont proches de 1.0, plus la performance est bonne.")

    except FileNotFoundError:
        print(f"\n❌ ERREUR: Le fichier '{data_file}' est introuvable.")
    except Exception as e:
        print(f"\n❌ ERREUR FATALE (probablement liée à NumPy MINGW): Le script a échoué.")
        print("Détails: Si les erreurs NumPy persistent, essayez de mettre à jour ou de réinstaller NumPy.")