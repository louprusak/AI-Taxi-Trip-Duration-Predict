#!/usr/bin/env python
# coding: utf-8

# # **Projet Prédiction de temps de trajet d'un taxi à New York | RUSAK & LEVY-VALENSI G3**

    # ## 1) Identifier et définir le problème

        # Importation des librairies nécessaires :

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from math import sin, cos, sqrt, atan2, radians
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE


    # ## 2) Compréhension des données :

        # ### Création du dataset à partir du fichier test.csv

test = pd.read_csv('./test.csv')
test


        # ### Création du dataset à partir du fichier train.csv

train = pd.read_csv('./train.csv')
train


        # ### Quelle est la taille de la base de données

print("Nombre de lignes du data-set test :",test.shape[0])

print("Nombre de colonnes du data-set test :",test.shape[1])

print("Nombre de lignes du data-set train :",train.shape[0])

print("Nombre de colonnes du data-set train :",train.shape[1])


        # ### Les données comprennent-elles des caractéristiques pertinentes pour ma problèmatique ?

list(test.columns)


        # ### Quels sont les types de données

test.dtypes

train.dtypes


        # ### Statistiques de base pour les attributs clés

test.describe()

train.describe()


        # ### Valeurs manquantes

test.isna().sum()

train.isna().sum()


    # ## 3) Préparation des données

        # ### Conversion des dates et horaires

        #Conversion du format des dates
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

        #Création des éléments séparés des dates
train['month'] = train.pickup_datetime.dt.month
train['weekday'] = train.pickup_datetime.dt.weekday
train['hour'] = train.pickup_datetime.dt.hour
train['minute'] = train.pickup_datetime.dt.minute
train['year'] = train.pickup_datetime.dt.year

test['month'] = test.pickup_datetime.dt.month
test['weekday'] = test.pickup_datetime.dt.weekday
test['hour'] = test.pickup_datetime.dt.hour
test['minute'] = test.pickup_datetime.dt.minute
test['year'] = test.pickup_datetime.dt.year

print('Année :',train['year'].sort_values().unique())
print('Heures :',train['hour'].sort_values().unique())
print('Mois :',train['month'].sort_values().unique())
print('Jours :',train['weekday'].sort_values().unique())

        # ### Affluence par heure

        # Le temps de trajet dépend-il du moment de la journée ?

range = train['hour'].sort_values().unique()
plt.hist(train['hour'],range,edgecolor='red',color='orange')
plt.xlabel("Heures")
plt.ylabel("Nombre de voyages")
plt.title('Nombre de voyages par tranches horaire')


        # ### Affluence par mois

        # Le temps de trajet peut-il dépendre du mois de l'année ?

range = train['month'].sort_values().unique()
range2 = [1,2,3,4,5,6,7]
plt.hist(train['month'],range2,edgecolor='red',color='orange')
plt.xlabel("Mois")
plt.ylabel("Nombre de voyages")
plt.title('Nombre de voyages par mois')


        # ### Affluence par jour

        # Le temps de trajet dépend-il du jour de la semaine ?

range = train['weekday'].sort_values().unique()
range2 = [0,1,2,3,4,5,6,7]
plt.hist(train['month'],range2,edgecolor='red',color='orange')
plt.xlabel("Jours de la semaine")
plt.ylabel("Nombre de voyages")
plt.title('Nombre de voyages par jours de la semaine')


        # Fonction de calcul de la distance parcourue en fonction des coordonnées.

def distance(long1, long2, lat1, lat2):

  # Approximation du royon de la Terre en km
  Rayon = 6373.0

  # Conversion des latitudes en radians
  lat1 = np.radians(lat1)
  lat2 = np.radians(lat2)

  # Conversion des longitudes en radians
  long1 = np.radians(long1)
  long2 = np.radians(long2)

  # Calcul de la différence entre latitudes et entre longitudes
  dif_long = long2 - long1
  dif_lat = lat2 - lat1

  a = np.sin(dif_lat /2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dif_long /2)**2

  c = 2 * np.arctan(np.sqrt(a), np.sqrt(1 - a))

  distance = Rayon * c

  return distance


        # Calcul des distances pour les deux datasets :

test['distance'] = distance(test['pickup_longitude'].values,
                            test['dropoff_longitude'].values,
                            test['pickup_latitude'].values,
                            test['dropoff_latitude'].values)

train['distance'] = distance(train['pickup_longitude'].values,
                            train['dropoff_longitude'].values,
                            train['pickup_latitude'].values,
                            train['dropoff_latitude'].values)


        # Nettoyage des données abbérantes :

test = test[(test.distance < 200)]
train = train[(train.distance < 200)]


        # ### Calcul de la vitesse

train['vitesse'] = train.distance / ((train.trip_duration/3600))



    # # 4) Résolution du problème



          # ## Régression par arbres de décisions


        # ### Préparation du DataSet d'entrées et de sorties pour l'apprentissage :

        # Le modèle choisi içi nécessite que les données soit divisées dans des ensemble de tests et d'apprentissage comme nous le faisons ci-dessous :

        # #### Choisissez ces colonnes si vous souhaitez que l'étude prenne en compte seulement les coordonnées et la distance :

columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude', 'distance']


# #### Choisissez ces colonnes si vous souhaitez que l'étude prenne en compte les affluences des heures, jours et mois :

#columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
#                    'dropoff_latitude', 'month' , 'hour','minute',
#                       'weekday', 'distance']


# Ne demande pas beaucoup de temps ni de ressources
temp = train[0:10000]
X_train = temp[columns]
y_train = temp['trip_duration']

#Si vous voulez une étude la plus complète possible, choisissez les instructions suivantes et commentez celle qui
#demandent moins de ressources
# DEMANDE BEAUCOUP DE TEMPS ET DE RESSOURCES

#X_train = train[columns]
#y_train = train['trip_duration']


        # ### Prévisualisation du DataSet d'entrées pour l'apprentissage :

        # Ce DataSet contient les principales données nécessaires à la prédiction du temps de trajet, deux types de DataSet d'entrée ont retenu notre attention comme vous pourrez le voir ci-dessus.

X_train


        # ### Prévisualisation du DataSet de sorties attendues pour l'apprentissage :

        # Ce DataSet doit comporter les résultats et donc les sorties attendues du modèle, soit les temps de trajets.

y_train


        # ### Création du modèle de régression :

        # On utilise içi un modèle de régression par arbres de décisions
model = RandomForestRegressor() 


        # ### Application du modèle aux données :

        # On donne a la fonction de création du modèle les DataSets d'entrées et de sorties pour l'apprentissage.
model.fit(X_train, y_train)


        # ### Préparation du DataSet d'entrées pour le prédiction :

# Ne demande pas beaucoup de temps ni de ressources
temptest = test[0:10000]
X_test = temptest[columns]

#Si vous voulez une étude la plus complète possible, choisissez les instructions suivantes et commentez celle qui
#demandent moins de ressources
# DEMANDE PLUS DE TEMPS ET DE RESSOURCES

#X_test = test[columns]


        # ### Prévisualisation du DataSet d'entrées pour l'apprentissage :

X_test


        # ### Exécution de la fonction de prédiction :

y_pred = model.predict(X_test)


        # ### Vérification des tailles du DataSet obtenu :

X_test.index.shape, y_pred.shape


        # ### Création du DataSet Submission :

        # Ouverture du fichier zip contenant le modèle de DataSet submission :
submission = pd.read_csv('./sample_submission.zip')

# A choisir si vous avez opté pour l'étude incomplète :
tempsub = submission[0:10000]
tempsub['trip_duration'] = y_pred
tempsub.head(10)

# A choisir si vous avez opté pour l'étude complète :
#submission['trip_duration'] = y_pred
#submission.head(10)


        # ### Création du fichier de résultats :

# A choisir si vous avez opté pour l'étude incomplète :
tempsub.to_csv('submission.csv', index=False)

# A choisir si vous avez opté pour l'étude complète :
#submission.to_csv('submission.csv', index=False)


        # Le fichier submission.csv contenant les temps de trajet prédits à partir des données du fichier test.csv est maintenant crée dans votre répertoire !
