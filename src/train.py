#---------------------------------------------------------------------------------------------------#
# File name: train.py                                                                               #
# Autor: Chrissi2802                                                                                #
# Created on: 01.09.2023                                                                            #
# Content: This file provides functionalities for training ML models.                               #
#---------------------------------------------------------------------------------------------------#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split, LearningCurveDisplay
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, f1_score, log_loss
from sklearn.metrics import precision_score, recall_score, jaccard_score, roc_auc_score, confusion_matrix
import mlflow
import warnings


sns.set_style("darkgrid")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


class MP_Models:
    """Mobile Price Models
    """

    def __init__(self, dict_data, dict_params_user) -> None:
        """Constructor

        Args:
            dict_data (dict): Dictionary with the data
            dict_params_user (dict): Dictionary with the user parameters
        """

        self.dict_data = dict_data
        self.dict_params_user = dict_params_user

        # Models
        self.dict_models = {
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
        }

        # Trained models
        self.dict_models_trained = {key: None for key in self.dict_models.keys()}

        # Hyperparameter grid for all models
        with open(dict_params_user["prefix"] + "/src/train.yaml", "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Hyperparameter grid for all models
        self.dict_hyperparams = {key: yaml_data[key] for key in self.dict_models.keys()}
        
        # Best Hyperparameter for the models
        self.dict_best_hyperparams = {key: None for key in self.dict_models.keys()}

        # Prepare the data
        self.prepare_data()

        # Run through all models
        for model in self.dict_models.keys():
            print("Model:", model)
            # Check if the model is already in mlflow
            series_runs = self.check_if_experiment_in_mlflow(model)

            if series_runs is not None:
                self.load_model_params_from_mlflow(model, series_runs)
            else:
                print("   No hyperparameters for the " + model + " model were found in mlflow!")
                self.search_best_hyperparams(model)
                self.train(model)

            print()

    def prepare_data(self) -> None:
        """Prepare the data.
        """

        # Train test split (train test = tt)
        if self.dict_params_user["test_size"] != 0.0:
            self.dict_data["train_tt_x"], self.dict_data["test_tt_x"], \
            self.dict_data["train_tt_y"], self.dict_data["test_tt_y"] = train_test_split(
                self.dict_data["train_fe_x" if self.dict_params_user["extended_features"] else "train_x"],
                self.dict_data["train_y"],
                test_size=self.dict_params_user["test_size"],
                random_state=self.dict_params_user["random_state"],
            )
        else:
            self.dict_data["train_tt_x"] = self.dict_data["train_fe_x" if self.dict_params_user["extended_features"] else "train_x"]
            self.dict_data["train_tt_y"] = self.dict_data["train_y"]
            self.dict_data["test_tt_x"] = None
            self.dict_data["test_tt_y"] = None

    def search_best_hyperparams(self, model) -> None:
        """Search the best hyperparameters for the models.

        Args:
            model (str): Name of the model
        """        

        print("   Search the best hyperparameters for the " + model + " model.")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", self.dict_models[model]())
        ])

        dict_hyperparams_grid = self.dict_hyperparams[model]

        # dict_hyperparams_grid key starts with clf__
        dict_hyperparams_grid = {
            "clf__" + key: value for key, value in dict_hyperparams_grid.items()
        }

        # Using crossvalidation
        repeated_kfolds = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

        # Search for the best hyperparameters
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=dict_hyperparams_grid,
            scoring="accuracy",
            n_jobs=-1,
            cv=repeated_kfolds,
            verbose=1,
            error_score="raise",
            return_train_score=False,
        )

        search.fit(self.dict_data["train_tt_x"], self.dict_data["train_tt_y"])
        self.dict_best_hyperparams[model] = search.best_params_
        #print(self.dict_best_hyperparams[model])

    def check_if_experiment_in_mlflow(self, model) -> pd.Series:
        """Check if the experiment is already in mlflow.

        Args:
            model (str): Name of the model

        Returns:
            series_runs (pd.Series): Series with the information about the mlflow run
        """

        # Get the experiment ID based on the experiment name
        self.experiment = mlflow.get_experiment_by_name(self.dict_params_user["mlflow_experiment"])

        if self.experiment is None:
            return None
        else:
            df_runs = mlflow.search_runs(experiment_ids=[self.experiment.experiment_id])
        
        # Convert the column tags.mlflow.log-model.history to a list with dictionaries
        df_runs["tags.mlflow.log-model.history"] = df_runs["tags.mlflow.log-model.history"].apply(lambda x: json.loads(x))
        
        # Use only lines where the parameters match, sort by accuracy and end_time
        df_r = df_runs[
            (df_runs["params.random_state"] == str(self.dict_params_user["random_state"])) &
            (df_runs["params.test_size"] == str(self.dict_params_user["test_size"])) &
            (df_runs["params.extended_features"] == str(self.dict_params_user["extended_features"])) &
            (df_runs["params.model_name"] == model)
        ].sort_values(by=["metrics.accuracy", "end_time"], ascending=[False, False])

        # Check if there are any results
        if df_r.shape[0] == 0:
            return None
        else:
            # Use only the first / best model
            series_runs = df_r.iloc[0, :].dropna()
            return series_runs

    def load_model_params_from_mlflow(self, model, series_runs) -> None:
        """Load the model from mlflow.

        Args:
            model (str): Name of the model
            series_runs (pd.Series): Series with the information about the mlflow run
        """

        run_id = series_runs["run_id"]
        
        # Only use data when columns start with params.
        series_runs = series_runs[series_runs.index.str.startswith("params.")]

        # Select only relevant columns / hyperparameters for this model
        dict_params_hyperparameter_list = list(self.dict_hyperparams[model].keys())
        dict_params_hyperparameter_list = ["params.clf__" + item for item in dict_params_hyperparameter_list]
        series_runs = series_runs[series_runs.index.isin(dict_params_hyperparameter_list)]

        # Remove params. from index
        series_runs.index = series_runs.index.str.replace("params.clf__", "")

        # Convert the values to the correct type
        series_runs = series_runs.apply(lambda x: int(x) if str(x).lstrip("-").isdigit() else float(x) 
                          if str(x).replace(".", "", 1).isdigit() else str(x)).to_dict()
        
        # Check if the keys for the hyperparameters are the same
        if set(series_runs.keys()) == set(self.dict_hyperparams[model].keys()):
            # Add params.clf__ to the keys
            series_runs = {"params.clf__" + key: value for key, value in series_runs.items()}

            # Save the best hyperparameters
            self.dict_best_hyperparams[model] = series_runs

            # Load the model from mlflow
            self.dict_models_trained[model] = mlflow.pyfunc.load_model("runs:/" + run_id + "/" + model)

            print("   Model " + model + " were found in mlflow!")
        else:
            raise Exception("The hyperparameters for the model " + model + " are not the same!")

    def train(self, model) -> None:
        """Train the model.

        Args:
            model (str): Name of the model
        """
        
        print("   Train and evaluate the " + model + " model.")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", self.dict_models[model]())
        ])

        self.pipeline.set_params(**self.dict_best_hyperparams[model])
        self.pipeline.fit(self.dict_data["train_tt_x"], self.dict_data["train_tt_y"])

        if self.dict_params_user["test_size"] != 0.0:
            self.y_pred = self.pipeline.predict(self.dict_data["test_tt_x"])
            self.y_pred_proba = self.pipeline.predict_proba(self.dict_data["test_tt_x"])
            self.evaluate_clf_model()
            self.plot_confusion_matrix()

        self.plot_learning_curve()
        self.save_mlflow(model)
        self.dict_models_trained[model] = self.pipeline

    def evaluate_clf_model(self) -> None:
        """Evaluate the model with different metrics.
        """

        self.dict_metrics = {
            "accuracy": accuracy_score(self.dict_data["test_tt_y"], self.y_pred),
            "balanced_accuracy": balanced_accuracy_score(self.dict_data["test_tt_y"], self.y_pred),
            "top_k_accuracy": top_k_accuracy_score(self.dict_data["test_tt_y"], self.y_pred_proba),
            "f1": f1_score(self.dict_data["test_tt_y"], self.y_pred, average="weighted"),
            "log_loss": log_loss(self.dict_data["test_tt_y"], self.y_pred_proba),
            "precision": precision_score(self.dict_data["test_tt_y"], self.y_pred, average="weighted"),
            "recall": recall_score(self.dict_data["test_tt_y"], self.y_pred, average="weighted"),
            "jaccard": jaccard_score(self.dict_data["test_tt_y"], self.y_pred, average="weighted"),
            "roc_auc": roc_auc_score(self.dict_data["test_tt_y"], self.y_pred_proba, average="weighted", multi_class="ovo"),
        }

    def plot_confusion_matrix(self) -> None:
        """Create a visualization of the confusion matrix.
        """
    
        self.fig_confusion_matrix, ax = plt.subplots()
        cm = confusion_matrix(self.dict_data["test_tt_y"], self.y_pred, normalize="true")
        sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    def plot_learning_curve(self) -> None:
        """Create a visualization of the learning curve.
        """
    
        self.fig_learning_curve, ax = plt.subplots()
        display = LearningCurveDisplay.from_estimator(
            self.pipeline, 
            self.dict_data["train_fe_x" if self.dict_params_user["extended_features"] else "train_x"], 
            self.dict_data["train_y"], 
            scoring="accuracy",
            score_name="Accuracy",
            score_type="both",
            line_kw={"marker": "o"},
            train_sizes=np.linspace(0.1, 1.0, 10),
            std_display_style="fill_between",
            n_jobs=-1,
            random_state=self.dict_params_user["random_state"],
            shuffle=True,
            verbose=0,
            ax=ax,
        )

        self.fig_learning_curve = display.figure_
        ax.set_title("Learning curve")
        ax.legend(["Training accuracy", "Test accuracy"], loc="best")

    def save_mlflow(self, model) -> None:
        """Save all relevant information in mlflow.

        Args:
            model (str): Name of the model
        """

        mlflow.set_experiment(self.dict_params_user["mlflow_experiment"])

        with mlflow.start_run():
            
            # Dataset
            mlflow.log_input(mlflow.data.from_pandas(
                pd.concat([self.dict_data["train_tt_x"], self.dict_data["train_tt_y"]], axis=1)), 
                context="train",
            )

            if self.dict_params_user["test_size"] != 0.0:
                mlflow.log_input(mlflow.data.from_pandas(
                    pd.concat([self.dict_data["test_tt_x"], self.dict_data["test_tt_y"]], axis=1)), 
                    context="test",
                )

            # Model
            mlflow.sklearn.log_model(self.pipeline, model)
            mlflow.log_params(self.dict_best_hyperparams[model])

            if self.dict_params_user["test_size"] != 0.0:
                # Metrics
                mlflow.log_metrics(self.dict_metrics)

                # Figures
                mlflow.log_figure(self.fig_confusion_matrix, "confusion_matrix.png")
            
            mlflow.log_figure(self.fig_learning_curve, "learning_curve.png")

            # User parameters
            mlflow.log_params(self.dict_params_user)
            mlflow.log_param("model_name", model)

        mlflow.end_run()


if __name__ == "__main__":

    from data import MP_Dataset
    
    # Information
    with open("info.yaml", "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    dict_params_user = yaml_data["dict_params_user"]
    
    # Data
    MPD = MP_Dataset(prefix=dict_params_user["prefix"])
    dict_data = MPD.get_data()

    # Training
    MPM = MP_Models(dict_data, dict_params_user)

