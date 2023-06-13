from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold


# Ce fichier contient les fonctions necessaires à l'étape 11

def afficher_evaluation(y_test, y_pred):
    print("+ Evaluation du modèle :")
    print("   Accuracy:", accuracy_score(y_test, y_pred))
    print("   Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("   Precision:", precision_score(y_test, y_pred))
    print("   Recall:", recall_score(y_test, y_pred))
    print("   F1-score:", f1_score(y_test, y_pred))


def validation_croisee(model, X, y, nb_folds=5):
    print("+ Validation croisée :")
    kfold = KFold(n_splits=nb_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print("   Scores:", scores)
    print("   Mean Accuracy:", scores.mean())
