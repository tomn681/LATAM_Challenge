import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        """
        DelayModel constructor.
        
        Selected Model: 
            XGBoost with Feature Importance and with Balance
        
        Reason:
            As we want to maximize the correctness of the model while
            maximizing the rightness of it, we prefer paying more attention
            to precision and recall values, specially the combination f1 for 
            class delayed = 1.
            In this way, we exclude models without balance. 
            Considering the data scientist's comments, we exclude base models.
            Finally we choose xg_boost with balance and feature importance due
            to slightly better f1-score.
        """
        
        self._model = None
        self.set_model()
        
        self.fitted_model = False
        
        self.top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        
    def set_model(
        self,
        target: pd.DataFrame = None
    ) -> None:
        """
        Set model using training data scaling.
        
        Args:
            target (pd.DataFrame, optional): Training targets for scaling 
                                             (y_true). Defaults to no scaling.
        
        Returns:
            None
        """
        scale = None
        
        if target is not None:
            n_y0 = len(target[target["delay"] == 0])
            n_y1 = len(target[target["delay"] == 1])
            
            scale = n_y0/n_y1
        
        self._model = xgb.XGBClassifier(random_state=1, 
                                        learning_rate=0.01, 
                                        scale_pos_weight = scale)
                                        
        return
                                        
    def load_from_weights(
        self,
        path: str
    ) -> None:
        """
        Load a pretrained model from weights.
        
        Args:
            path (str): Weights location
        
        Returns:
            None
        """            
        self._model.load_model(path)
        
        return
    
    def save_model(
        self,
        model_name: str = "XGBoost",
        path: str = "./"
    ) -> None:
        """
        Save the current model.
        
        Args:
            model_name (str, optional): Model Name
            path (str, optional): Saving Path
        
        Returns:
            None
        """
        model_name = f"{model_name}_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}.txt"
        self._model.save_model(os.path.join(path, model_name))
        
        return
                                        
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """        
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        
        # Train/Test Mode
        if target_column:
            data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            target = data[[target_column]]
            return features[self.top_10_features], target
            
        # If incomplete data
        add_cols = [col for col in self.top_10_features if col not in features.columns]
        features.loc[:, add_cols] = 0
        
        # Predict Default
        return features[self.top_10_features]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self.set_model(target) #For Scaling
        
        x_train = features
        y_train = target
        
        self._model.fit(x_train, y_train)
        self.fitted_model = True
        
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if not self.fitted_model:
            self.load_from_weights("./challenge/MODEL.txt")
        
        y_preds = self._model.predict(features)
        y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]
        
        return y_preds
                
    @staticmethod
    def get_min_diff(
        data: pd.DataFrame,
        columns: List[str] = ['Fecha-O', 'Fecha-I']
    ) -> np.float64:
        """
        Get difference in minutes.
        
        Args:
            data (pd.DataFrame): raw data.
            columns (List[str], optional): Columns to substract. 
                                           Default: ['Fecha-O', 'Fecha-I']
            
        Returns:
            (np.float64): Difference in minutes.
        """
        assert len(columns) == 2, \
                        f'get_min_diff recieves exactly 2 arguments, not {len(columns)}'
        fecha_o = datetime.strptime(data[columns[0]], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data[columns[1]], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
            
if __name__ == "__main__":
    data_path = "../data/data.csv"
    data = pd.read_csv('../data/data.csv')
    
    model = DelayModel()
    features, target = model.preprocess(data, "delay")
    
    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]
    
    print(isinstance(features, pd.DataFrame))
    print(features.shape[1] == len(FEATURES_COLS))
    print(set(features.columns) == set(FEATURES_COLS))

    print(isinstance(target, pd.DataFrame))
    print(target.shape[1] == len(TARGET_COL))
    print(set(target.columns) == set(TARGET_COL))
    
    print(data["OPERA"])
    
    model.fit(features, target)
    
    predictions = model.predict(features)
