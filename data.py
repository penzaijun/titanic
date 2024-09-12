import csv
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class Passenger:
    PassengerId:int
    #numerical
    Age:float
    Fare:float
    SibSp:int
    Parch:int
    #categorical    
    Pclass:int      # 1 or 2 or 3
    Sex:str         # 'male' or 'female'
    Embarked:str    # 'S' or 'C' or 'Q'
    #str 
    Name:str
    Cabin:str
    Ticket:str
    #label
    Survived:int=-1

def load_data(norm=True):
    df_train=pd.read_csv('./train.csv')
    df_test=pd.read_csv('./test.csv')
    train_num=len(df_train)
    df=pd.concat([df_train,df_test],axis=0)
    print(df.info())
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Age']=df['Age'].fillna(-df['Age'].max())
    if norm:
        columns=['Age','Fare','SibSp','Parch']
        for column in columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    train=[Passenger(**row) for _,row in df[:train_num].iterrows()]
    test=[Passenger(**row) for _,row in df[train_num:].iterrows()]
    return train,test

def visualize(df,column='Fare'):
    sns.histplot(df[column], bins=20)
    plt.title('title')
    plt.xlabel('count')
    plt.ylabel('value')
    plt.show()

Pclass2onehot=lambda x:[1,0,0] if x==1 else [0,1,0] if x==2 else [0,0,1]
Sex2onehot=lambda x:[1,0] if x=='male' else [0,1]
Embarked2onehot=lambda x:[1,0,0] if x=='S' else [0,1,0] if x=='C' else [0,0,1]

def load_csv(path,pre_process=True) -> list[Passenger]:
    data=csv.DictReader(open(path,'r'))
    passenger_list=[]
    header_int=['SibSp','Parch','Pclass']
    header_float=['Age','Fare']
    for row in data:
        if pre_process:
            try:
                if row['Age']=='':row['Age']=-100
                if row['Fare']=='':row['Fare']=7.2292
                for h in header_int:
                    row[h]=int(row[h])
                for h in header_float:
                    row[h]=float(row[h])
                if 'Survived' in row: row['Survived']=int(row['Survived'])
            except Exception as e:
                print(f'ERROR in load data {row}')
        passenger_list.append(Passenger(**row))    
    return passenger_list

def write_csv(data:list[Passenger],path='result.csv'):
    w=csv.DictWriter(open(path,'w',newline=''),fieldnames=['PassengerId','Survived'])
    w.writeheader()
    w.writerows([{'PassengerId':d.PassengerId,'Survived':d.Survived} for d in data])

class TitanicDataset(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        self.data=self.data2feature(data)
    
    def data2feature(self,data:list[Passenger]):
        out=[]
        for d in data:
            feature=[d.Age,d.Fare,d.Parch,d.SibSp]
            feature.extend(Pclass2onehot(d.Pclass))
            feature.extend(Sex2onehot(d.Sex))
            feature.extend(Embarked2onehot(d.Embarked))
            out.append((np.array(feature,dtype=np.float32),np.array(d.Survived)))
        self.feature_dim=int(len(feature))
        return out
    
    def __getitem__(self, index):
        feature=torch.tensor(self.data[index][0])
        label=torch.tensor(self.data[index][1]).long()
        return feature,label
    
    def __len__(self):
        return len(self.data)