import csv as csv


print("hello world")

with open('../ressources/car_insurance.csv', 'r') as f:
    # Créer un objet csv à partir du fichier
    obj = csv.reader(f)

    for ligne in obj:
        print(ligne)