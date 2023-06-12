import pandas as pd
import matplotlib.pyplot as pp
from etape_quatre import *
from etape_cinq import *

# import du dataframe
dataframe = pd.read_csv("../ressources/car_insurance.csv")


# analyse des données
etape_quatre(dataframe)
# correction des données
etape_cinq(dataframe)


