# ====================================================================
# utils.py - Fonctions et Classes pour l'Atelier d'Apprentissage Supervisé
# ====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import warnings
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    make_scorer
)

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=10000, suppress=True)

# ====================================================================
# I. CLASSES DE TRANSFORMATION PERSONNALISÉES (POUR PIPELINE)
# ====================================================================

class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Transformateur pour appliquer l'ACP et concaténer les composantes."""
    
    def __init__(self, n_components=3, random_state=1):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
    
    def fit(self, X, y=None):
        self.pca.fit(X)
        return self
    
    def transform(self, X, y=None):
        X_pca = self.pca.transform(X)
        if isinstance(X, list):
            X = np.array(X)
        return np.concatenate((X, X_pca), axis=1)
        
class FeatureSelectorByIndex(BaseEstimator, TransformerMixin):
    """Transformateur pour sélectionner les variables par indices (Q5)."""
    
    def __init__(self, indices):
        self.indices = np.array(indices).astype(int) 
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[:, self.indices]

# ====================================================================
# II. FONCTIONS D'ÉVALUATION ET DE COMPARAISON (Q2, Q10)
# ====================================================================

def evaluate_model(model, X_test, y_test, model_name="Modèle", mode='test'):
    """
    Évalue un modèle (Matrice de confusion, ROC, Score final: (Acc + Recall) / 2) (Q2).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # Utilisation du rappel (Recall) comme meilleur critère pour le scoring
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) 
    auc_score = roc_auc_score(y_test, y_proba)
    score_final = (accuracy + recall) / 2

    print(f"--- Résultats {model_name} ({mode}) ---")
    print(f"Matrice de confusion:\n{cm}")
    print(f"Accuracy: {accuracy:.4f} | Rappel: {recall:.4f} | Score final (Acc+Recall)/2: {score_final:.4f}")

    # Courbe ROC 
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend()
    plt.show()

    return {
        'accuracy': accuracy,
        'recall': recall,
        'auc': auc_score,
        'score_final': score_final 
    }

def run_classifiers_train_test(classifiers, X_train, y_train, X_test, y_test, mode='standard'):
    """
    Entraîne et compare les classifieurs sur les bases Train/Test (Q2, Q3).
    """
    print(f"\n--- Comparaison des classifieurs ({mode}) ---")
    meilleur_score = -1
    meilleur_nom = ""
    meilleur_modele = None

    for nom, modele in classifiers.items():
        modele.fit(X_train, y_train)
        res = evaluate_model(modele, X_test, y_test, nom, mode=mode)
        
        if res['score_final'] > meilleur_score:
            meilleur_score = res['score_final']
            meilleur_nom = nom
            meilleur_modele = modele

    print(f"\n🏆 MEILLEUR: {meilleur_nom} (Score final: {meilleur_score:.4f})")
    return meilleur_nom, meilleur_modele

def run_classifiers_cv(X_original, Y):
    """Compare plusieurs classifieurs par 10-fold CV (Q10)."""
    
    print("\n" + "="*80)
    print("QUESTION 10: COMPARAISON ROBUSTE (10-FOLD CROSS-VALIDATION)")
    print("="*80)
    
    # Prétraitement des configurations de données pour CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    pca = PCA(n_components=3, random_state=1)
    X_pca_augmented = np.concatenate([X_scaled, pca.fit_transform(X_scaled)], axis=1)

    datasets = {'Original': X_original, 'Normalisé': X_scaled, 'Normalisé + ACP': X_pca_augmented}
    
    clfs = {
        'CART': DecisionTreeClassifier(random_state=1), 
        'ID3 (Entropy)': DecisionTreeClassifier(criterion='entropy', random_state=1), 
        'Decision Stump': DecisionTreeClassifier(max_depth=1, random_state=1), 
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'MLP (20-10)': MLPClassifier(hidden_layer_sizes=(20, 10), random_state=1, max_iter=500),
        'Bagging (200)': BaggingClassifier(DecisionTreeClassifier(random_state=1), n_estimators=200, random_state=1, n_jobs=-1),
        'AdaBoost (200)': AdaBoostClassifier(n_estimators=200, random_state=1),
        'Random Forest (200)': RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1),
        'XGBoost (200)': XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=1, n_jobs=-1) 
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    best_global_score = -1
    best_global_clf_name = ""
    best_global_ds_name = ""

    for ds_name, ds_X in datasets.items():
        print(f"\n--- Jeu de Données: {ds_name} ---")
        
        for clf_name, clf in clfs.items():
            start_time = time.time()
            
            cv_acc = cross_val_score(clf, ds_X, Y, cv=kf, scoring='accuracy', n_jobs=-1)
            cv_auc = cross_val_score(clf, ds_X, Y, cv=kf, scoring='roc_auc', n_jobs=-1)
            cv_precision = cross_val_score(clf, ds_X, Y, cv=kf, scoring='precision_weighted', n_jobs=-1)
            cv_score_final = (cv_acc + cv_precision) / 2
            
            mean_acc = np.mean(cv_acc)
            std_acc = np.std(cv_acc)
            mean_auc = np.mean(cv_auc)
            std_auc = np.std(cv_auc)
            mean_score_final = np.mean(cv_score_final)
            std_score_final = np.std(cv_score_final)
            exec_time = time.time() - start_time
            
            print(f"  {clf_name:<20} | ACC: {mean_acc:.4f} +/- {std_acc:.4f} | AUC: {mean_auc:.4f} +/- {std_auc:.4f} | SCORE_F: {mean_score_final:.4f} +/- {std_score_final:.4f} | Time: {exec_time:.2f}s")
            
            if mean_score_final > best_global_score:
                best_global_score = mean_score_final
                best_global_clf_name = clf_name
                best_global_ds_name = ds_name
                
    print("\n" + "="*80)
    print("🏆 IDENTIFICATION DU MEILLEUR ALGORITHME GLOBAL (POST-CV)")
    print("="*80)
    print(f"Meilleur Algorithme Global: {best_global_clf_name} sur {best_global_ds_name} (Score: {best_global_score:.4f})")
    
    return best_global_clf_name, best_global_ds_name

# ====================================================================
# III. FONCTIONS DE SÉLECTION ET TUNING (Q5, Q6)
# ====================================================================

def importance_des_variables(Xtrain, Ytrain, nom_cols):
    """Calcule l'importance relative des variables (Q5)."""
    clf = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1) 
    clf.fit(Xtrain, Ytrain)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    features = np.array(nom_cols)

    print("\nVariables triées par importance :")
    print(features[sorted_idx])
    plt.figure(figsize=(10, 8))
    padding = np.arange(Xtrain.shape[1]) + 0.5 
    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center', color='darkgreen')
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Importance Relative")
    plt.title("Importance des Variables (Random Forest)")
    plt.gca().invert_yaxis() 
    plt.show()

    return sorted_idx, features

def selection_nombre_optimal_variables(Xtrain, Xtest, Ytrain, Ytest, sorted_idx, meilleur_algo_name, meilleur_algo_params):
    """Détermine le nombre optimal de variables à conserver (Q5)."""
    print(f"\n🔄 Détermination du nombre optimal de variables (avec {meilleur_algo_name})...")

    # Initialiser le meilleur modèle (MLP ou autre)
    if meilleur_algo_name == 'MLP':
        best_clf = MLPClassifier(random_state=1, max_iter=1000, **meilleur_algo_params)
    elif meilleur_algo_name.startswith('Random Forest'):
        best_clf = RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=-1, **meilleur_algo_params)
    else:
        best_clf = DecisionTreeClassifier(random_state=1)
    
    scores = np.zeros(Xtrain.shape[1])
    
    for f in np.arange(0, Xtrain.shape[1]):
        X1_f = Xtrain[:, sorted_idx[:f+1]]
        X2_f = Xtest[:, sorted_idx[:f+1]] 
        
        # Réinitialiser/recréer le classifieur à chaque itération
        clf_iteration = best_clf.__class__(**best_clf.get_params())
        clf_iteration.fit(X1_f, Ytrain)
        
        Y_pred = clf_iteration.predict(X2_f)
        scores[f] = np.round(accuracy_score(Ytest, Y_pred), 4)
        
    optimal_features_count = np.argmax(scores) + 1
    max_accuracy = scores[optimal_features_count - 1]
    
    print(f"Meilleure Accuracy: {max_accuracy:.4f} (avec {optimal_features_count} variables)")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, Xtrain.shape[1] + 1), scores, marker='o', linestyle='-', color='blue')
    plt.plot(optimal_features_count, max_accuracy, 'ro', markersize=8, label=f'Optimal: {optimal_features_count} vars ({max_accuracy:.4f})')
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Accuracy")
    plt.title(f"Évolution de l'Accuracy en fonction des variables ({meilleur_algo_name})")
    plt.xticks(np.arange(1, Xtrain.shape[1] + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return optimal_features_count

def score_acc_prec(Y_true, Y_pred, **kwargs):
    """Critère pour GridSearchCV: (accuracy + precision pondérée) / 2."""
    acc = accuracy_score(Y_true, Y_pred)
    prec = precision_score(Y_true, Y_pred, average='weighted', zero_division=0) 
    return (acc + prec) / 2

scorer_acc_prec = make_scorer(score_acc_prec, greater_is_better=True)

def recherche_meilleurs_parametres(X_train_selected, Y_train, base_clf, param_grid):
    """Recherche les meilleurs hyperparamètres (Q6, Q10)."""
    
    print("\n🔄 Recherche des meilleurs hyperparamètres (GridSearchCV)...")

    kf_grid = KFold(n_splits=5, shuffle=True, random_state=0)
    
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring=scorer_acc_prec,
        cv=kf_grid,
        verbose=0,
        n_jobs=-1 
    )

    start_time = time.time()
    grid_search.fit(X_train_selected, Y_train)
    end_time = time.time()
    
    best_params = grid_search.best_params_
    
    print(f"\n✅ Recherche terminée en {end_time - start_time:.2f} secondes.")
    print(f"   Meilleurs paramètres: {best_params}")
    
    return grid_search.best_estimator_, best_params

def creation_pipeline(X_train_data, Y_train_data, best_k_indices, final_clf, pipeline_filename):
    """Crée, entraîne et sauvegarde le pipeline final (Q7, Q10)."""
    
    print(f"\n🔄 Création et Entraînement du pipeline vers {pipeline_filename}...")
    
    pipeline_steps = [
        ('scaler', StandardScaler()),
        ('pca_augmenter', FeatureAugmenter(n_components=3, random_state=1)),
        ('feature_selector', FeatureSelectorByIndex(best_k_indices)),
        ('final_classifier', final_clf)
    ]

    pipeline = Pipeline(pipeline_steps)
    pipeline.fit(X_train_data, Y_train_data) 
    
    with open(pipeline_filename, 'wb') as file:
        pickle.dump(pipeline, file)
        
    print(f"✅ Pipeline créé, entraîné, et sauvegardé.")
    
    return pipeline

# ====================================================================
# IV. FONCTIONS D'ORCHESTRATION (Q8, Q10)
# ====================================================================

def pipeline_generation_train_test_split(df, X_train, Y_train, X_test, Y_test):
    """Orchestration complète pour la première partie (Q8)."""
    
    print("\n" + "#"*80)
    print("ORCHESTRATION DU PIPELINE (Q1 à Q8)")
    print("#"*80)
    
    # --- 1. Préparation des données (Normalisation & ACP) ---
    print("\n[Étape 1] Préparation des données: Normalisation + ACP...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=3, random_state=1)
    pca.fit(X_train_scaled)
    X_train_with_pca = np.concatenate([X_train_scaled, pca.transform(X_train_scaled)], axis=1)
    X_test_with_pca = np.concatenate([X_test_scaled, pca.transform(X_test_scaled)], axis=1)
    
    nom_cols_pca = list(df.columns[:-1]) + ['PC1', 'PC2', 'PC3']
    
    # --- 2. Importance et Sélection (Q5) ---
    print("[Étape 2] Sélection de variables (RF + MLP)...")
    sorted_idx, _ = importance_des_variables(X_train_with_pca, Y_train, nom_cols_pca)
    
    # Paramètres par défaut/simulés pour le MLP de la sélection 
    mlp_selection_params = {'hidden_layer_sizes': (40, 20), 'max_iter': 500, 'random_state': 1}
    k_optimal = selection_nombre_optimal_variables(
        X_train_with_pca, X_test_with_pca, Y_train, Y_test, sorted_idx, 'MLP', mlp_selection_params
    )
    best_k_indices = sorted_idx[:k_optimal] 
    
    # --- 3. Tuning du MLP (Q6) ---
    print("[Étape 3] Tuning du MLP (GridSearchCV)...")
    X_train_selected_for_tuning = X_train_with_pca[:, best_k_indices]
    base_mlp = MLPClassifier(random_state=1, max_iter=1000)
    param_grid_mlp = {'hidden_layer_sizes': [(30, 15), (40, 20), (50, 25)], 'activation': ['tanh', 'relu'], 'alpha': [0.0001, 0.001, 0.01]}
    tuned_mlp, best_mlp_params = recherche_meilleurs_parametres(X_train_selected_for_tuning, Y_train, base_mlp, param_grid_mlp)

    # --- 4. Création et Sauvegarde du Pipeline (Q7/Q8) ---
    print("[Étape 4] Création et Entraînement du pipeline...")
    final_pipeline = creation_pipeline(
        X_train, Y_train, best_k_indices, tuned_mlp, "credit_scoring_pipeline.pkl"
    )
    
    return final_pipeline

def pipeline_generation_cv(df, X_data, Y_data, best_cv_clf_name, best_cv_clf_params, k_optimal, X_test, Y_test):
    """Orchestration finale du pipeline basée sur les résultats CV (Q10)."""
    
    print("\n" + "#"*80)
    print(f"ORCHESTRATION FINALE DU PIPELINE (Q10: {best_cv_clf_name})")
    print("#"*80)
    
    # --- 1. Préparation des données pour l'étape d'importance ---
    print("\n[Étape 1] Préparation des données: Normalisation + ACP...")
    temp_scaler = StandardScaler()
    X_scaled = temp_scaler.fit_transform(X_data)
    temp_pca = PCA(n_components=3, random_state=1)
    temp_pca.fit(X_scaled)
    X_train_with_pca = np.concatenate([X_scaled, temp_pca.transform(X_scaled)], axis=1)
    
    # --- 2. Importance des variables (pour obtenir l'ordre) ---
    print("[Étape 2] Sélection de variables (RF)...")
    nom_cols_pca = list(df.columns[:-1]) + ['PC1', 'PC2', 'PC3']
    clf_rf_importances = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
    clf_rf_importances.fit(X_train_with_pca, Y_data)
    sorted_idx = np.argsort(clf_rf_importances.feature_importances_)[::-1]
    
    best_k_indices = sorted_idx[:k_optimal] 
    
    # --- 3. Définir le classifieur final tuné ---
    print(f"[Étape 3] Définition du classifieur final ({best_cv_clf_name})...")
    if best_cv_clf_name.startswith('Random Forest'):
        final_clf = RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=-1, **best_cv_clf_params)
    elif best_cv_clf_name.startswith('XGBoost'):
        final_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1, n_estimators=200, **best_cv_clf_params)
    elif best_cv_clf_name.startswith('MLP'):
        final_clf = MLPClassifier(random_state=1, max_iter=1000, **best_cv_clf_params)
    else:
        final_clf = DecisionTreeClassifier(random_state=1)
        
    # --- 4. Création et Sauvegarde du Pipeline Final ---
    final_pipeline = creation_pipeline(
        X_data, Y_data, best_k_indices, final_clf, "final_production_pipeline_cv.pkl"
    )
    
    # --- 5. Évaluation du pipeline CV final sur le jeu de test ---
    Y_pred_pipe = final_pipeline.predict(X_test)
    Y_proba_pipe = final_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Évaluation du Pipeline Final (sur jeu de test) ---")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred_pipe):.4f}")
    print(f"AUC: {roc_auc_score(Y_test, Y_proba_pipe):.4f}")
    
    return final_pipeline

# ====================================================================
# V. FONCTIONS PARTIE II : DONNÉES HÉTÉROGÈNES
# ====================================================================

def traitement_donnees_numeriques(df, col_num):
    """Prépare et nettoie les données numériques (II.1)."""
    print("\n" + "#"*80)
    print("PARTIE II - QUESTION 1: DONNÉES NUMÉRIQUES SEULES")
    print("#"*80)
    
    # Conversion en numpy array pour l'indexation
    data = df.values 
    X_full = data[:, :-1] 
    Y_full = data[:, -1]
    
    # Extraire les colonnes numériques, remplacer '?' par NaN, et typer en float
    X_num = np.copy(X_full[:, col_num])
    # Assurez-vous que les '?' sont remplacés par np.nan
    X_num[X_num == '?'] = np.nan
    X_num = X_num.astype(float)
    
    # Supprimer les individus contenant des nan (basé sur df temporaire pour utiliser la fonction any)
    df_temp = pd.DataFrame(X_num)
    mask = df_temp.isnull().any(axis=1) # Masque des lignes contenant des NaN
    X_cleaned = X_num[~mask]
    Y_cleaned = Y_full[~mask]
    
    Y_bin = Y_cleaned.astype(int) 

    print(f"Taille après nettoyage: {X_cleaned.shape[0]} individus ({X_cleaned.shape[1]} variables)")
    
    X_train_num, X_test_num, Y_train_num, Y_test_num = train_test_split(
        X_cleaned, Y_bin, test_size=0.5, random_state=1
    )
    
    return X_train_num, X_test_num, Y_train_num, Y_test_num

def traitement_donnees_heterogenes_imputation(X_full, Y_full, col_num, col_cat):
    """Traite les données hétérogènes avec imputation (II.2)."""
    print("\n" + "#"*80)
    print("PARTIE II - QUESTION 2: DONNÉES HÉTÉROGÈNES ET MANQUANTES")
    print("#"*80)
    
    # --- 1. Préparation et imputation des variables numériques (Mean) ---
    X_num = np.copy(X_full[:, col_num])
    X_num[X_num == '?'] = np.nan
    X_num = X_num.astype(float)
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_num_imputed = imp_num.fit_transform(X_num)

    # --- 2. Préparation et imputation des variables catégorielles (Most Frequent) ---
    X_cat = np.copy(X_full[:, col_cat])

    X_cat[X_cat == '?'] = np.nan

    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_cat_imputed = imp_cat.fit_transform(X_cat)

    # --- 3. Encodage One-Hot des variables catégorielles ---
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_bin = ohe.fit_transform(X_cat_imputed)

    # --- 4. Construction et Normalisation du jeu de données final ---
    X_combined = np.concatenate((X_num_imputed, X_cat_bin), axis=1)
    Y_bin = Y_full.astype(int)
    
    X_train_comb, X_test_comb, Y_train_comb, Y_test_comb = train_test_split(
        X_combined, Y_bin, test_size=0.5, random_state=1
    )
    
    scaler_final = StandardScaler()
    X_train_final = scaler_final.fit_transform(X_train_comb)
    X_test_final = scaler_final.transform(X_test_comb)
    
    print(f"Dimensions finales (Entraînement): {X_train_final.shape}")
    
    return X_train_final, X_test_final, Y_train_comb, Y_test_comb