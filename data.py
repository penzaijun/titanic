import csv
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re 
from sklearn.preprocessing import MinMaxScaler

def load_data():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test1.csv')
    df_train['Dataset'] = 'Train'
    df_test['Dataset'] = 'Test'
    df = pd.concat([df_train, df_test], ignore_index=True)

    #feature engingeering
    df['Cabin'] = df['Cabin'].notna()

    def extract_title(name):
        match = re.search(r'([A-Za-z]+)\.', name)
        return match.group(1)
    
    df['Title'] = df['Name'].apply(extract_title)
    
    grouped_titles = {
        "Mr": ["Mr"],
        "Mrs": ["Mrs","Mme"],
        "Master":["Master"],
        "Miss": ["Miss","Ms","Mlle"],
        "Crew": ["Dr", "Rev", "Col", "Major", "Capt"],
        "Noble": ["Lady", "Sir", "Countess", "Jonkheer", "Don", "Dona" ]
    }

    def categorize_title(title):
        for group, titles in grouped_titles.items():
            if title in titles:
                return group

    df["TitleGroup"] = df["Title"].apply(categorize_title)

    ticket_counts = df["Ticket"].value_counts()
    df["Alone"] = df["Ticket"].map(ticket_counts == 1)

    # Preprocessing
    # Fill missing values
    df["Age"] = df["Age"].fillna(df.groupby("TitleGroup")["Age"].transform("mean"))
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # One-hot encoding
    df=df.join(pd.get_dummies(df['Pclass'],prefix='Pclass').astype(float))
    df=df.join(pd.get_dummies(df['Sex'],prefix='Sex').astype(float))
    df=df.join(pd.get_dummies(df['Embarked'],prefix='Embarked').astype(float))
    df=df.join(pd.get_dummies(df['TitleGroup'],prefix='TitleGroup').astype(float))
    df=df.join(pd.get_dummies(df['Alone'],prefix='Alone').astype(float))
    df=df.join(pd.get_dummies(df['Cabin'],prefix='Cabin').astype(float))

    df.drop(['Pclass', 'Sex', 'Embarked','TitleGroup','Title','Name',"Ticket",'Alone','Cabin'], axis=1, inplace=True)

    # norm
    df['Fare'] = np.log(df['Fare']+1)
    scaler = MinMaxScaler()
    df[['Age', 'SibSp', 'Parch', 'Fare']] = scaler.fit_transform(df[['Age', 'SibSp', 'Parch', 'Fare']])

    # seperate train and test
    df_train = df[df['Dataset'] == 'Train'].drop('Dataset', axis=1)
    df_test = df[df['Dataset'] == 'Test'].drop('Dataset', axis=1)

    # Select relevant features and target variable for training
    features = [col for col in df.columns if col not in ["PassengerId", "Survived","Dataset"]]
    x = df_train[features]
    y = df_train['Survived']
    passenger_ids = df_test['PassengerId']
    x_test = df_test[features]
    y_test=df_test['Survived'] if 'Survived' in df_test else None
    return x,y,x_test,passenger_ids,y_test

def save_result(model, predictions,passenger_ids):
    output = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    output['Survived'] = output['Survived'].astype(int)
    # Save the results to a CSV file
    output_file = './results/result_' + model + '.csv'
    output.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")

class TitanicDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x.values, dtype=torch.float32)  
        self.y = None if y is None else torch.tensor(y.values, dtype=torch.long)  
        self.feature_dim = self.x.shape[1]  
    
    def __len__(self):
        return len(self.x)  
    
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx], torch.tensor(0, dtype=torch.long)  
        return self.x[idx], self.y[idx]  