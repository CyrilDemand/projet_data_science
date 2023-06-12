import pandas as pd

dataframe = pd.read_csv("../ressources/car_insurance.csv")

# Supposons que vous ayez un DataFrame appelé "dataframe" avec deux colonnes "colonne1" et "colonne2"
# Utilisez la méthode corr() sur les deux colonnes pour calculer leur corrélation

# ================================
# Age
# ================================

correlation = dataframe['age'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre age et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['gender'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre gender et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['driving_experience'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre driving_experience et outcome :", correlation)
# ================================
# Age
# ================================

correlation = dataframe['education'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre education et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['income'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre income et outcome :", correlation)
# ================================
# Age
# ================================

correlation = dataframe['credit_score'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre credit_score et outcome :", correlation)
# ================================
# Age
# ================================

correlation = dataframe['vehicle_ownership'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre vehicle_ownership et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['vehicle_year'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre vehicle_year et outcome :", correlation)
# ================================
# Age
# ================================

correlation = dataframe['married'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre married et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['children'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre children et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['postal_code'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre postal_code et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['annual_mileage'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre annual_mileage et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['vehicle_type'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre vehicle_type et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['speeding_violations'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre speeding_violations et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['duis'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre duis et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['age'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre age et outcome :", correlation)

# ================================
# Age
# ================================

correlation = dataframe['past_accidents'].corr(dataframe['outcome'])

# Affichez la corrélation entre les deux colonnes
print("Corrélation entre past_accidents et outcome :", correlation)