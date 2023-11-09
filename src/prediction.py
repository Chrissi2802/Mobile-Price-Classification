#---------------------------------------------------------------------------------------------------#
# File name: prediction.py                                                                          #
# Autor: Chrissi2802                                                                                #
# Created on: 12.10.2023                                                                            #
# Content: This file provides functionalities for predictions.                                      #
#---------------------------------------------------------------------------------------------------#


import pandas as pd
import mlflow
import json

try:
    from data import feature_engineering
except:
    from src.data import feature_engineering


class MP_Prediction:
    """Mobile Price Prediction
    """

    def __init__(self, dict_params_user) -> None:
        """Constructor

        Args:
            dict_params_user (dict): Dictionary with the user parameters
        """
                
        self.dict_params_user = dict_params_user

        # Find and load the best model
        self.__find_load_best_model()

    def __find_load_best_model(self) -> None:
        """Search the model with the best accuracy and the smallest test size. 
           Then load this model with the test size 0.0.
        """
        
        # Get the experiment ID based on the experiment name
        experiment = mlflow.get_experiment_by_name(self.dict_params_user["mlflow_experiment"])
        
        # Get all runs of the experiment
        df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Convert the columns to a list with dictionaries and to float
        df_runs["tags.mlflow.log-model.history"] = df_runs["tags.mlflow.log-model.history"].apply(lambda x: json.loads(x))
        df_runs["params.test_size"] = df_runs["params.test_size"].astype(float)

        # Find the best model
        best_model_name = df_runs[ 
            (df_runs["params.test_size"] > 0.0)
        ].sort_values(by=["metrics.accuracy", "end_time"], ascending=[False, False])["params.model_name"].iloc[0]

        # Get the best model
        self.series_runs = df_runs[
            (df_runs["params.test_size"] == 0.0) &
            (df_runs["params.model_name"] == best_model_name)
        ].sort_values(by=["metrics.accuracy", "end_time"], ascending=[False, False]).iloc[0, :].dropna()

        # Load the best model
        self.model = mlflow.sklearn.load_model("runs:/" + self.series_runs["run_id"] + "/" + best_model_name)

    def predict(self, df) -> dict:
        """Predict the price range.

        Args:
            df (pandas DataFrame): DataFrame with the data

        Returns:
            prediction (dict): Predicted price range and the probability for each class
        """
        
        # Feature engineering if extended features are used
        if self.series_runs["params.extended_features"]:
            df = feature_engineering(df)

        prediction = {
            "pred": self.model.predict(df),
            "pred_proba": self.model.predict_proba(df),
        }

        return prediction


if __name__ == '__main__':
    
    import yaml
    from data import MP_Dataset

    # Information
    with open("info.yaml", "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    dict_params_user = yaml_data["dict_params_user"]

    MPD = MP_Dataset(prefix=dict_params_user["prefix"])
    dict_data = MPD.get_data()
    columns = dict_data["train_x"].columns

    # Test data --> 1
    df = pd.DataFrame(
        [[842.0, 0.0, 2.2, 0.0, 1.0, 0.0, 7.0, 0.6, 188.0, 2.0, 2.0, 20.0, 756.0, 2549.0, 9.0, 7.0, 19.0, 0.0, 0.0, 1.0]],
        columns=columns,
    )

    MPP = MP_Prediction(dict_params_user)
    
    prediction = MPP.predict(df)
    print(prediction)
    
