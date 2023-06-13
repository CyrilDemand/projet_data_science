import pandas as pd
import matplotlib.pyplot as pp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from etape_quatre import *
from etape_cinq import *
from src.etape_six import etape_six

# import du dataframe depuis le fichier csv
dataframe = pd.read_csv("../ressources/car_insurance.csv")

# ETAPE 4 : Analyse des données
# etape_quatre(dataframe)

# ETAPE 5 : Correction des données
cleaned_dataframe = etape_cinq(dataframe)

# ETAPE 6 : recherche de corrélations
# etape_six(cleaned_dataframe)

# ETAPE 7 : Création des jeux de données d'entrainement et de test
X = cleaned_dataframe.drop('outcome', axis=1)
y = cleaned_dataframe['outcome']
# la taille du jeu de test est de 20% de la taille du jeu de données total (test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train est le jeu de données d'apprentisage, avec ses étiquettes y_train
# X_test est le jeu de données de test, avec ses étiquettes y_test

# ETAPE 8 : Entrainement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# ETAPE 9 : Evaluation du modèle

# On utilise le modèle pour faire des prédictions sur le jeu de test
y_pred = model.predict(X_test)

# Différentes mesures permettent d'évaluer la qualité du modèle
# l'accuracy mesure la proportion de prédictions correctes parmi toutes les prédictions
print("Accuracy:", accuracy_score(y_test, y_pred))
# la matrice de confusion permet de visualiser les vrais positifs, les vrais négatifs, les faux positifs et les faux négatifs.
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
# la précision mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives
print("Precision:", precision_score(y_test, y_pred))
# le rappel (recall) mesure la proportion de vrais positifs correctement identifiés parmi tous les vrais positifs.
print("Recall:", recall_score(y_test, y_pred))
# F-mesure (F1-score) : Combinaison de la précision et du rappel pour obtenir une mesure globale.
print("F1-score:", f1_score(y_test, y_pred))

# ETAPE 10 : Améloration de l'évaluation du modèle

# On peut améliorer l'évaluation du modèle en utilisant la validation croisée
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print("Scores:", scores)
print("Mean Accuracy:", scores.mean())