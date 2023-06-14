import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def etape_six(dataframe):
    # on utilise la méthode corr() pour verifier la corrélation entre l'outcome et et chacuns autres colonnes
    # le but est d'identifier les colonnes qui ont le plus d'influence sur l'outcome

    # plus la valeur est proche de 1 ou -1 plus la corrélation est forte :
    # Une valeur de 1 indique une corrélation positive parfaite, ce qui signifie que les deux variables sont linéairement liées de manière positive.
    # Une valeur de -1 indique une corrélation négative parfaite, ce qui signifie que les deux variables sont linéairement liées de manière négative (quand l'une augmente, l'autre diminue).

    # les affichages suivants sont triés par corrélation, du plus corrélé au moins corrélé (en valeur absolue)

    print("Corrélation entre driving_experience et outcome :", dataframe['driving_experience'].corr(dataframe['outcome']))
    print("Corrélation entre age et outcome :", dataframe['age'].corr(dataframe['outcome']))
    print("Corrélation entre vehicle_ownership et outcome :", dataframe['vehicle_ownership'].corr(dataframe['outcome']))
    print("Corrélation entre past_accidents et outcome :", dataframe['past_accidents'].corr(dataframe['outcome']))
    print("Corrélation entre credit_score et outcome :", dataframe['credit_score'].corr(dataframe['outcome']))
    print("Corrélation entre vehicle_year et outcome :", dataframe['vehicle_year'].corr(dataframe['outcome']))
    print("Corrélation entre speeding_violations et outcome :", dataframe['speeding_violations'].corr(dataframe['outcome']))
    print("Corrélation entre children et outcome :", dataframe['children'].corr(dataframe['outcome']))
    print("Corrélation entre duis et outcome :", dataframe['duis'].corr(dataframe['outcome']))
    print("Corrélation entre annual_mileage et outcome :", dataframe['annual_mileage'].corr(dataframe['outcome']))
    print("Corrélation entre postal_code et outcome :", dataframe['postal_code'].corr(dataframe['outcome']))
    print("Corrélation entre gender et outcome :", dataframe['gender'].corr(dataframe['outcome']))
    print("Corrélation entre education et outcome :", dataframe['education'].corr(dataframe['outcome']))
    print("Corrélation entre income et outcome :", dataframe['income'].corr(dataframe['outcome']))
    print("Corrélation entre vehicle_type et outcome :", dataframe['vehicle_type'].corr(dataframe['outcome']))

    # on peut voir que les données qui ont le plus d'influence sur l'outcome sont surtout liées à l'experience sur la route (driving_experience,age,vehicle_ownership)

    # on peut visualiser ces corrélations les plus importantes avec un graphique grace à la méthode scatter_matrix()
    pd.plotting.scatter_matrix(dataframe[['driving_experience','outcome']], figsize=(8, 8))
    # Calcul de la matrice de corrélation
    corr_matrix = dataframe.corr()['outcome'].to_frame()

    # Affichage de la matrice de corrélation sous forme de heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélation')
    plt.show()


    # on pourrait ici supprimer les colonnes qui n'ont presque pas d'influence sur l'outcome
    # dataframe.drop('vehicle_type', axis=1, inplace=True)

