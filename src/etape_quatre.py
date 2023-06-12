import pandas as pd
import matplotlib.pyplot as pp

def etape_quatre(dataframe):

    print(dataframe['postal_code'].value_counts())
    # ========================================================================================
    # VERIFICATION DES VALEURS NULLES
    # ========================================================================================

    # Comptage des valeurs non nulles pour chaque colonnes
    print(dataframe.count())
    # les colonnes credit_score & annual_mileage ont des valeurs manquantes
    # ces valeurs seront remplacées dans la partie 5


    # ========================================================================================
    # VERIFICATION DES VALEURS ABHÉRENTES (pour les valeurs numériques)
    # ========================================================================================

    # on peut visualiser une colonne avec matplotlib et la méthode hist()
    dataframe['age'].hist()
    pp.show()
    # on peut aussi utiliser value_counts() pour compter le nombre de fois qu'appraissent chaque valeur


    # VALEURS ABHERENTES IDENTIFIEES POUR LES COLONNES SUIVANTES :
    # - nombre d'enfants : 24,72 et 103
    print(dataframe['children'].value_counts())
    # - nombre d'exces de vitesses : 345,2867,6743,12035,41056
    print(dataframe['speeding_violations'].value_counts())

    # ========================================================================================
    # VERIFICATION DES VALEURS ABHÉRENTES (pour les valeurs catégoriques)
    # ========================================================================================

    # pour les valeurs non numériques, on peut utiliser la méthode isin() pour vérifier si les valeurs sont conformes

    # toutes les données sont comprises dans les valeurs 1,2,3 ou 4
    print( dataframe['age'].isin([0,1,2,3]).all() ) # => True
    # toutes les données sont comprises dans les valeurs 0 ou 1
    print( dataframe['gender'].isin([0,1]).all() ) # => True
    # toutes les données sont conformes
    print(dataframe['driving_experience'].isin(['0-9y','10-19y','20-29y','30y+']).all() ) # => True
    # toutes les données sont conformes
    print( dataframe['education'].isin(['none','high school','university']).all() ) # => True
    # toutes les données sont conformes
    print( dataframe['income'].isin(['poverty','working class','middle class','upper class']).all() ) # => True
    # toutes les données sont conformes (0.0 ou 1.0)
    print( dataframe['vehicle_ownership'].isin([0.0,1.0]).all() ) # => True
    # toutes les données sont conformes
    print( dataframe['vehicle_year'].isin(['before 2015','after 2015']).all() ) # => True
    # toutes les données sont conformes
    print( dataframe['married'].isin([0.0,1.0]).all() ) # => True
    # toutes les données sont conformes
    print( dataframe['vehicle_type'].isin(['sedan','sports car']).all() ) # => True