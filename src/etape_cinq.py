import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as pp
le = preprocessing.LabelEncoder()

def etape_cinq(dataframe):

    # ========================================================================================
    # SUPPRESSION DES COLONNES INUTILES A L'ETUDE
    # ========================================================================================

    # Id non necessaire à l'étude, on supprime la colonne
    dataframe = dataframe.drop('id', axis=1)


    # ========================================================================================
    # CORRECTION DES DONNEES MANQUANTES
    # ========================================================================================

    # 982 valeurs null pour le score de credit
    # On remplace les valeurs manquantes par la mediane de toutes les valeurs
    median = dataframe['credit_score'].median()
    dataframe['credit_score'].fillna(median, inplace=True)

    # 957 valeurs null pour le nombre de miles roulés par an
    # On remplace les valeurs manquantes par la mediane de toutes les valeurs
    median = dataframe['annual_mileage'].median()
    dataframe['annual_mileage'].fillna(median, inplace=True)


    # ========================================================================================
    # CORRECTION DES DONNEES ABHERENTES
    # ========================================================================================

    # NOMBRE D'ENFANTS
    # on remplace les valeurs supérieurs à un certain seuil par la mediane des valeurs
    median = dataframe['children'].median()
    # seuil fixé à 20 enfants
    dataframe.loc[dataframe['children'] > 20, 'children'] = median

    # NOMBRE D'EXCES DE VITESSE
    # on remplace les valeurs supérieurs à un certain seuil par la mediane des valeurs
    median = dataframe['speeding_violations'].median()
    # seuil fixé à 30 exces de vitesse
    dataframe.loc[dataframe['speeding_violations'] > 30, 'speeding_violations'] = median


    # ========================================================================================
    # TRANSFORMATION DES DONNEES CATEGORIQUES EN DONNEES NUMERIQUES
    # ========================================================================================

    # Driving experience
    le.fit(['0-9y','10-19y','20-29y','30y+'])
    dataframe['driving_experience'] = le.transform(dataframe['driving_experience'])

    # Education
    le.fit(['none','high school','university'])
    dataframe['education'] = le.transform(dataframe['education'])

    # Income
    le.fit(['poverty','working class','middle class','upper class'])
    dataframe['income'] = le.transform(dataframe['income'])
    print(dataframe['income'])

    # Vehicle year
    le.fit(['before 2015','after 2015'])
    dataframe['vehicle_year'] = le.transform(dataframe['vehicle_year'])

    # Postal code
    le.fit(['10238','32765','92101','21217'])
    dataframe['postal_code'] = le.transform(dataframe['postal_code'])

    # Vehicle type
    le.fit(['sedan','sports car'])
    dataframe['vehicle_type'] = le.transform(dataframe['vehicle_type'])


    # TODO : Normalisation en tableau numpy avec des valeurs entre 0 et 1