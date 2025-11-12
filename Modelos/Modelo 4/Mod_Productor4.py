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
from umap import UMAP


class Mod_Productor():
    def __init__(self, data:pd.DataFrame, target:str, test_size:float=0.2, random_state:int=42, n_components:int=2):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.X = self.data.drop(columns=[self.target])
        self.n_components = n_components
        self.y = self.data[self.target]
        self.model = None
        self.X_test = None
        self.y_test = None
        self.umap = None
    
    def _split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_model(self):
        ## First apply UMAP to reduce dimensionality
        print(f"Original feature count: {self.X.shape[1]}")
        
        umap = UMAP(n_components=self.n_components, random_state=self.random_state)
        X_umap = umap.fit_transform(self.X)
        
        print(f"Features after UMAP: {X_umap.shape[1]}")
        
        ## Now apply SMOTE to balance the UMAP-transformed data
        print(f"\nClass distribution before SMOTE:")
        print(f"  Class 0 (Unhealthy): {sum(self.y == 0)}")
        print(f"  Class 1 (Healthy): {sum(self.y == 1)}")
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_umap, self.y)
        
        print(f"\nClass distribution after SMOTE:")
        print(f"  Class 0 (Unhealthy): {sum(y_balanced == 0)}")
        print(f"  Class 1 (Healthy): {sum(y_balanced == 1)}")

        # Store the balanced data for training
        self.X_train, self.y_train = X_balanced, y_balanced

        # Define parameter grid for hyperparameter tuning
        # Note: No UMAP in pipeline since we already applied it
        param_grid = [
            # L1 regularization options
            {
                'penalty': ['l1'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],
                'class_weight': ['balanced']  # Data already balanced by SMOTE
            },
            # L2 regularization options
            {
                'penalty': ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'class_weight': ['balanced']  # Data already balanced by SMOTE
            },
            # Elasticnet option (saga only)
            {
                'penalty': ['elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],
                'l1_ratio': [0.2, 0.5, 0.8],
                'class_weight': ['balanced']  # Data already balanced by SMOTE
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

        # Create model directly (no pipeline needed since UMAP already applied)
        base_model = LogisticRegression(max_iter=1000, random_state=self.random_state)

        # Use multiple scoring with refit on the primary metric
        print("\nStarting GridSearchCV...")
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring=scoring,
            refit='accuracy',  # Which metric to use for selecting best model
            return_train_score=True,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        
        # Store the best model, UMAP transformer, and results
        self.model = best_model
        self.umap = umap  # Store UMAP for later use in prediction
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
        
        if self.umap is None:
            raise ValueError("UMAP transformer not found. Call train_model() first.")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply UMAP transformation first
        X_umap = self.umap.transform(X)
        
        # Then predict
        return self.model.predict(X_umap)

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
        
        if self.umap is None:
            raise ValueError("UMAP transformer not found. Call train_model() first.")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply UMAP transformation first
        X_umap = self.umap.transform(X)
        
        # Then predict probabilities
        return self.model.predict_proba(X_umap)

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