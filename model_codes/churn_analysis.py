import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import get_logger
from utils.file_handler import load_csv,save_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
logger=get_logger(name='churn_analysis')

logger.info('starting churn analysis')

def preprocessing(df):
    try:
        print(df.isna().sum())
        print(df.shape)
        df=df.dropna()
        print(df.shape)
        print(df.dtypes)
        logger.debug('null values drop')
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        categorical_cols =['user_segment','platform']

        for col in categorical_cols:
            le=LabelEncoder()
            df[col] = le.fit_transform(df[col])
        logger.info('categorical values encoding')
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].mean())
            scaler=StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            
        logger.info('numerical values scaling')

        le = LabelEncoder()
        df['churned'] = le.fit_transform(df['churned'])         
        df.drop_duplicates(inplace=True)
        print(df.head())
        logger.debug('duplicate values drop')
        return df
    except Exception as e:
        logger.error('preprocessing error: {}'.format(e))
        raise 

def feature_engineering(df):
    try:
            
        logger.info('feature engineering started')
        print(df.columns.tolist())
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        df['engagement_ratio'] = df['monthly_viewing_hours'] / df['monthly_cost']
        df['is_inactive'] = (df['days_since_last_watch'] > 14).astype(int)
        df['cost_per_hour'] = df['monthly_cost'] / (df['monthly_viewing_hours'] + 1)  # +1 to avoid div by zero
        df['tenure_bin'] = pd.cut(df['tenure_months'], bins=[0, 6, 12, 24, 60], labels=['0-6m','6-12m','1-2y','2-5y'])
        df['is_satisfied'] = (df['satisfaction_score'] >= 3).astype(int)
        logger.info('feature engineering completed')
        df.drop(['tenure_months','satisfaction_score'],axis=1,inplace=True)
        df.drop(['monthly_viewing_hours', 'monthly_cost', 'days_since_last_watch'], axis=1, inplace=True)
        return df
    except Exception as e:
        logger.error('feature engineering error: {}'.format(e))
        raise 

def train_test_splitter(df):
    try:
            
        logger.info('train test split started')
        x=df.drop(['churned'],axis=1)
        y=df['churned']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logger.info('train test split completed')
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error('train test split error: {}'.format(e))
        raise 
    
def model_selection(x_train,y_train,x_test,y_test):
    try:
        models={
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42 ),
        'Linear Regression': LinearRegression(),
        'lightGBM': LGBMRegressor(),
        'CatBoost': CatBoostClassifier(),
        'XGBoost': XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        
        }
    

        model_reports = []  # 👈 To store all model results

        for name, model in models.items():
            logger.info(f'{name} model training started')
            model.fit(x_train, y_train)
            logger.info(f'{name} model training completed')

            y_pred = model.predict(x_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)  # ✅ for classification metrics

            logger.info(f'{name} model evaluation started')
            print(f'{name} model:')

            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred_binary)
            prec = precision_score(y_test, y_pred_binary)
            rec = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)
            roc_auc = roc_auc_score(y_test, y_pred_binary)
            cm = confusion_matrix(y_test, y_pred_binary)

            print(f'Mean Squared Error: {mse}')
            print(f'R2 Score: {r2}')
            print(f'Confusion Matrix:\n{cm}\n')

            logger.info(f'{name} model trained and evaluated successfully')

            # Model summary
            summary = model.get_params()
            
            # Append metrics + key params to report list
            model_reports.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'MSE': mse,
                'R2': r2,
                **summary  # Unpack model parameters (optional, can skip if too long)
            })

        # ✅ Convert all collected data to a DataFrame
        report_df = pd.DataFrame(model_reports)

        # ✅ Save to CSV
        # summary.to_csv('model_codes/all_model_summary.csv', index=False)
        # logger.info('Model summary saved to CSV')
        report_df.to_csv("model_codes/all_model_reports.csv", index=False)
        logger.info('All model reports saved to CSV')
    except Exception as e:
        logger.error('model selection error: {}'.format(e))
        raise

def hyperparameter_tuning(x_train, y_train, x_test, y_test):
    try:
        # Define base model
        model = LGBMRegressor()

        # Define hyperparameters (based on your best performing config)
        params = {
            "boosting_type": ["gbdt"],
            "n_estimators": [100],
            "learning_rate": [0.1],
            "colsample_bytree": [1.0],
            "importance_type": ["split"],
            "min_child_samples": [20],
            "min_child_weight": [0.001],
            "min_split_gain": [0.0],
            "num_leaves": [31],
            "subsample": [1.0],
            "subsample_for_bin": [200000],
            "reg_alpha": [0.0],
            "reg_lambda": [0.0]
        }

        # Grid Search
        grid_search = GridSearchCV(model, params, cv=5, scoring='r2')
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Prediction
        y_pred = best_model.predict(x_test)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        # Evaluation
        print(f'Best Parameters: {best_params}')
        print(f'Best Model: {best_model}')
        print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
        print(f'R2 Score: {r2_score(y_test, y_pred)}')
        # print(f'Accuracy Score: {accuracy_score(y_test, y_pred_binary)}')
        # print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_binary)}')
        # print(f'Classification Report:\n{classification_report(y_test, y_pred_binary)}')

        logger.info('Hyperparameter tuning completed successfully')
        return best_model, best_params

    except Exception as e:
        logger.error('Hyperparameter tuning error: {}'.format(e))
        raise


def save_model(model):
    try:
        import joblib
        joblib.dump(model, 'model_codes/churn_model.pkl')
        logger.info('Model saved successfully')
    except Exception as e:
        logger.error('Model saving error: {}'.format(e))
        raise 
    
def main():
    logger.info('Starting churn analysis')
    df=load_csv(r'C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\Streamlined_Netflix_Churn.csv')
    print(df.head())
    df=preprocessing(df)
    print(df.head())
    df=feature_engineering(df)
    
    # 1. Identify object or category columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # 2. Encode them numerically (Label Encoding for tree/logistic models)
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    print("Before split:")
    print(df['churned'].value_counts())

    x_train, x_test, y_train, y_test = train_test_splitter(df)
    best_model,best_params=hyperparameter_tuning(x_train,y_train,x_test,y_test)
    print(best_params)
    # model_selection(x_train,x_test,y_train,y_test)
    save_model(best_model)
    logger.info('Churn analysis completed successfully')
    # x_train, x_test, y_train, y_test = train_test_splitter(df)
    # # 
    # model=LGBMRegressor()
    # print(model.get_params)
    # model_selection(x_train,x_test,y_train,y_test)
    
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # df = pd.read_csv("model_codes/all_model_reports.csv")

    # # Plot Accuracy comparison
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=df, x="Model", y="Accuracy", palette="viridis")
    # plt.title("Model Accuracy Comparison")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()