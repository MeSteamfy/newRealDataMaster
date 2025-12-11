# Fichier principal

Le fichier principal est [celui ci](./newTPAlgoDataMaster.ipynb)

# Atelier Apprentissage Supervisé - Credit Scoring

Ce projet implémente un pipeline complet de Machine Learning pour le scoring de crédit, allant de l'analyse exploratoire à la mise en production via une API FastAPI.

## 📋 Prérequis

- **Python** (version 3.9 ou supérieure)
- **Poetry** (Gestionnaire de dépendances)

Si Poetry n'est pas installé sur votre machine :

```bash
# Windows : 
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
$Env:Path += ";$Env:APPDATA\Python\Scripts"
poetry --version

# Mac
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc
poetry --version

# Linux
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version


## Clonez le dépôt et accédez au dossier :

git clone git@github.com:MeSteamfy/newRealDataMaster.git
cd newRealDataMaster
```

## Installez l'environnement virtuel et les dépendances :

```bash
poetry lock
poetry install
```

## Lancez Jupyter via Poetry
```bash
poetry run jupyter notebook
```

Ouvrir et exécuter le notebook : <br>
    Dans l'interface web qui s'ouvre, cliquez sur le fichier **newTPAlgoDataMaster.ipynb**<br>
    Dans le menu du notebook, sélectionnez Kernel > Restart & Run All.<br>
    Cette étape va sauvegarder le fichier **credit_scoring_pipeline.pkl** sur votre disque.<br>

## Lancement de l'API

### Démarrez le serveur Uvicorn
```bash
poetry run uvicorn main:app --reload
```

L'API est maintenant accessible. Vous pouvez tester les endpoints via la documentation interactive :
    URL : http://127.0.0.1:8000/docs

## 🧪 Utilisation de l'API

L'API expose un endpoint de prédiction qui accepte les données d'un client bancaire et retourne une décision (Accordé/Refusé) avec les probabilités associées.

### Endpoint
`POST /predict/`

### Format des données (Input)
Le modèle attend une liste de **13 valeurs numériques** respectant l'ordre précis du dataset d'entraînement.

Voici la signification de chaque index dans la liste `features` :

| Index | Nom Variable | Description | Exemple |
| :--- | :--- | :--- | :--- |
| 0 | **Seniority** | Ancienneté professionnelle (années) | `9.0` |
| 1 | **Home** | Propriétaire (1) ou Locataire (0) etc. | `1.0` |
| 2 | **Time** | Durée du crédit (mois) | `60.0` |
| 3 | **Age** | Âge du client (années) | `30.0` |
| 4 | **Marital** | Statut marital (encodé) | `0.0` |
| 5 | **Records** | Incidents de paiement passés (0 ou 1) | `1.0` |
| 6 | **Job** | Type d'emploi (encodé) | `1.0` |
| 7 | **Expenses** | Dépenses mensuelles | `73.0` |
| 8 | **Income** | Revenus mensuels | `129.0` |
| 9 | **Assets** | Patrimoine / Actifs | `0.0` |
| 10 | **Debt** | Dette existante | `0.0` |
| 11 | **Amount** | Montant du crédit demandé | `800.0` |
| 12 | **Price** | Prix du bien à financer | `846.0` |

### Exemples de test (Curl)

**1. Profil Risqué (Refus probable)**
*Revenus faibles, pas d'actifs, gros montant demandé.*
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "Seniority": 0.0,
  "Home": 0.0,
  "Time": 48.0,
  "Age": 20.0,
  "Marital": 1.0,
  "Records": 1.0,
  "Job": 0.0,
  "Expenses": 100.0,
  "Income": 0.0,
  "Assets": 0.0,
  "Debt": 5000.0,
  "Amount": 1500.0,
  "Price": 1500.0
}'
```

**2. Profil Solide (Accord probable)**
*Revenus plus élevés (200), Actifs importants (5000), Dette nulle.*
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "Seniority": 15.0, "Home": 1.0, "Time": 36.0, "Age": 45.0, "Marital": 2.0,
  "Records": 0.0, "Job": 3.0, "Expenses": 50.0, "Income": 200.0,
  "Assets": 5000.0, "Debt": 0.0, "Amount": 500.0, "Price": 600.0
}'
```

## Connaitre la note du TP

Pour savoir la note du tp, il faut lancer la commande suivante:

```bash
poetry run python test_auc.py
```

Cela va prendre un peu de temps pour lancer un résultat, mais cette commande va nous donner un résultat du type:

```bash
--- 1. PRÉPARATION DES DONNÉES ---
Taille de l'échantillon final (lignes, colonnes) : (4375, 9)
Nombre d'exemples positifs (1) : 3159
Nombre d'exemples négatifs (0) : 1216
--------------------------------------------------
--- 2. EXÉCUTION DES CLASSIFIEURS ---

Modèle: Régression Logistique
  - Accuracy: 0.7845
  - AUC (Aire sous la courbe ROC): 0.7820

Modèle: K-Nearest Neighbors (K=5)
  - Accuracy: 0.7243
  - AUC (Aire sous la courbe ROC): 0.6784
--------------------------------------------------

✅ EXÉCUTION TERMINÉE.
Interprétation : Comparez les valeurs d'AUC. Plus elles sont proches de 1.0, plus la performance est bonne.
```

Comme dis dans le résultat de la fonction, plus le resultat des tests se rapprochent à 1, plus la note sera haute.