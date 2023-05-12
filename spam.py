import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# on lit le fichier de mail qui va nous ermettre d'entrer notre machine 
Data_mail = pd.read_csv("") # mettre son fichier

# remplace les valeurs nulles ou manquantes 
mail_data = Data_mail.where((pd.notnull(Data_mail)), "") 

# on selectionne les colonnes et lignes 
mail_data.loc[mail_data["Category"] == 'spam' , 'Category',] = 0
mail_data.loc[mail_data["Category"] == 'ham' , 'Category',] = 1

#on separe les données

X = mail_data["Message"]
Y = mail_data ["Category"]

# on divise les données en ensemble d'entraînement et ensemble de test 80/20
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=3)

# on transform les données en texte en vecteurs numériques
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='french',lowercase=True)


X_train_features = feature_extraction.fit_transform(X_train) # transformation en vecteurs de caractéristiques pondérés 
X_test_features = feature_extraction.transform(X_test) # on obtient des vecteurs de caractéristiques normalisés pour l'apprentissage 

#on convertit les valeurs de la variable en entiers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# création d'un modèle de régression logistique
model = LogisticRegression()
model.fit(X_train_features,Y_train)

# permet de prédire la catégorie des mails sur les données d'entraînement et d'obtenir la précision du modèle sur ces données.
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_traning_data = accuracy_score(Y_train, prediction_on_training_data)

#print('Accuracy on training data : ', accuracy_on_traning_data)

# permet de prédire la catégorie des mails sur les données de test et d'obtenir la précision du modèle sur ces données.
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)

#print('Accuracy on training data : ', accuracy_on_test_data)


input_mail = [""] # Entrez son mail

nouvelle_demande = feature_extraction.transform(input_mail)

prediction = model.predict(nouvelle_demande)
print(prediction)


if prediction[0] ==  1:
    print("bon mail")

else:
    print('spam mail')

 




  