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
        ## First apply PCA to reduce dimensionality
        print(f"Original feature count: {self.X.shape[1]}")
        
        pca = PCA(n_components=self.PCA_components)
        X_pca = pca.fit_transform(self.X)
        
        print(f"Features after PCA: {X_pca.shape[1]}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        
        ## Now apply SMOTE to balance the PCA-transformed data
        print(f"\nClass distribution before SMOTE:")
        print(f"  Class 0 (Unhealthy): {sum(self.y == 0)}")
        print(f"  Class 1 (Healthy): {sum(self.y == 1)}")
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_pca, self.y)
        
        print(f"\nClass distribution after SMOTE:")
        print(f"  Class 0 (Unhealthy): {sum(y_balanced == 0)}")
        print(f"  Class 1 (Healthy): {sum(y_balanced == 1)}")

        # Store the balanced data for training
        self.X_train, self.y_train = X_balanced, y_balanced

        # Define parameter grid for hyperparameter tuning
        # Note: No PCA in pipeline since we already applied it
        param_grid = [
            # L1 regularization options
            {
                'penalty': ['l1'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],
                'class_weight': [None]  # Data already balanced by SMOTE
            },
            # L2 regularization options
            {
                'penalty': ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'class_weight': [None]  # Data already balanced by SMOTE
            },
            # Elasticnet option (saga only)
            {
                'penalty': ['elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],
                'l1_ratio': [0.2, 0.5, 0.8],
                'class_weight': [None]  # Data already balanced by SMOTE
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

        # Create model directly (no pipeline needed since PCA already applied)
        base_model = LogisticRegression(max_iter=1000, random_state=self.random_state)

        # Use multiple scoring with refit on the primary metric
        print("\nStarting GridSearchCV...")
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring=scoring,
            refit='f1',  # Which metric to use for selecting best model
            return_train_score=True,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        
        # Store the best model, PCA transformer, and results
        self.model = best_model
        self.pca = pca  # Store PCA for later use in prediction
        self.grid_search_results = pd.DataFrame(grid_search.cv_results_)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print("Best parameters found: ", grid_search.best_params_)
        print("\nBest scores across all metrics:")
        print(f"  Accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
        print(f"  Precision: {grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]:.4f}")
        print(f"  Recall: {grid_search.cv_results_['mean_test_recall'][grid_search.best_index_]:.4f}")
        print(f"  F1-Score: {grid_search.cv_results_['mean_test_f1'][grid_search.best_index_]:.4f}")
        print(f"  ROC-AUC: {grid_search.cv_results_['mean_test_roc_auc'][grid_search.best_index_]:.4f}")

        return self.model

    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to make predictions on
            
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        if self.pca is None:
            raise ValueError("PCA transformer not found. Call train_model() first.")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply PCA transformation first
        X_pca = self.pca.transform(X)
        
        # Then predict
        return self.model.predict(X_pca)

    def predict_proba(self, X):
        """
        Predict class probabilities for X
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to make predictions on
            
        Returns:
        --------
        probabilities : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        if self.pca is None:
            raise ValueError("PCA transformer not found. Call train_model() first.")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply PCA transformation first
        X_pca = self.pca.transform(X)
        
        # Then predict probabilities
        return self.model.predict_proba(X_pca)

    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Test features
        y : array-like
            True labels
            
        Returns:
        --------
        metrics : dict
            Dictionary containing accuracy, precision, recall, f1_score
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        
        print("\nEvaluation Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        return metrics