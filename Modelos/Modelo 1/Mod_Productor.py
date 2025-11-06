# import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import Template
# Definición de modulos

import torch
import numpy  as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from imblearn.over_sampling  import SMOTE, ADASYN
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, recall_score, f1_score, precision_score
from sklearn.decomposition import PCA


class Mod_Productor():
    def __init__(self, data:pd.DataFrame, target:str, test_size:float=0.2, random_state:int=42, PCA_components:int=2):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.X = self.data.drop(columns=[self.target])
        self.PCA_components = PCA_components
        self.y = self.data[self.target]
        self.model = None
        self.X_test = None
        self.y_test = None
    
    def _split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_model(self):
        ## First use SMOTE to balance the data
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(self.X, self.y)

        # Create balanced dataframe
        df_balanced = pd.DataFrame(X_balanced, columns=self.X.columns)
        df_balanced.insert(0, 'Sana', y_balanced)

        # Data preparation for model production
        df_X = df_balanced.drop(columns=['Sana']).values
        df_y = df_balanced['Sana'].values

        # Split the balanced data
        self.X_train, self.y_train = df_X, df_y

        # Define a more comprehensive parameter grid for hyperparameter tuning
        param_grid = [
            # L1 regularization options
            {
                'PCA__n_components': [self.PCA_components],
                'model__penalty': ['l1'],
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__solver': ['liblinear', 'saga'],
                'model__class_weight': ['balanced']
            },
            # L2 regularization options
            {
                'PCA__n_components': [self.PCA_components],
                'model__penalty': ['l2'],
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__solver': ['liblinear', 'saga', 'lbfgs'],
                'model__class_weight': ['balanced']
            },
            # Elasticnet option (saga only)
            {
                'PCA__n_components': [self.PCA_components],
                'model__penalty': ['elasticnet'],
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__solver': ['saga'],
                'model__l1_ratio': [0.2, 0.5, 0.8],
                'model__class_weight': ['balanced']
            }
        ]
        # Define multiple scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        # Create a pipeline
        pipeline = Pipeline([
            ('PCA', PCA()),
            ('model', LogisticRegression(max_iter=1000))
        ])

        # Use multiple scoring with refit on the primary metric
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring=scoring,
            refit='f1',  # Which metric to use for selecting best model
            return_train_score=True
        )
        
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        
        # Store the best model and results
        self.model = best_model
        self.grid_search_results = pd.DataFrame(grid_search.cv_results_)
        
        print("Best parameters found: ", grid_search.best_params_)
        print("\nBest scores across all metrics:")
        print(f"  Accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
        print(f"  Precision: {grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]:.4f}")
        print(f"  Recall: {grid_search.cv_results_['mean_test_recall'][grid_search.best_index_]:.4f}")
        print(f"  F1-Score: {grid_search.cv_results_['mean_test_f1'][grid_search.best_index_]:.4f}")
        print(f"  ROC-AUC: {grid_search.cv_results_['mean_test_roc_auc'][grid_search.best_index_]:.4f}")

        return self.model