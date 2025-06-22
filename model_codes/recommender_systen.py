import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import get_logger
from utils.file_handler import load_csv,save_csv
from sklearn.preprocessing import LabelEncoder,StandardScaler

logger=get_logger(name='recommender syatem')

def preprocessing(df):
    try:
        print(df.head(10))
        print(df.isna().sum())
        print(df.dtypes)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        categorical_cols=df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col])
        
        numerical_cols=df.select_dtypes(include=['float64','int64']).columns
        for col in numerical_cols :
            scaler=StandardScaler()
            df[col]=scaler.fit_transform(df[[col]])
        print(df.head(10))
        return df
    except Exception as e:
        logger.error('error while preprocessing data,%s',e)
        raise
    
def feature_engineering(df):
    try:
        print(df.shape)
        df["engagement_ratio"] = df["watch_percentage"] / 100
        df["content_age"] = 2025 - df["release_year"]
        df["user_content_age_gap"] = df["user_age"] - df["content_age"]
        df["rating_per_minute"] = df["rating"] / (df["duration_minutes"] + 1e-3)
        df["watch_time_minutes"] = (df["duration_minutes"] * df["watch_percentage"]) / 100
        df["is_long_content"] = (df["duration_minutes"] > 90).astype(int)
        print(df.shape)
        print(df.head(10))
        return df
    except Exception as e:
        logger.error('feature engineering failing%s'.e)
        raise
    
def split_data(df):
    return    

def main():
    df=load_csv(r'C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\Full_Netflix_Dataset.csv')
    df=preprocessing(df)
    df=feature_engineering(df)
    

if __name__=='__main__':
    main()
    