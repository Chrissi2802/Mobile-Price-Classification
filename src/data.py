#---------------------------------------------------------------------------------------------------#
# File name: data.py                                                                                #
# Autor: Chrissi2802                                                                                #
# Created on: 01.09.2023                                                                            #
# Content: This file provides the data.                                                             #
#---------------------------------------------------------------------------------------------------#


import pandas as pd
import numpy as np


class MP_Dataset:
    """Mobile Price Dataset
    """

    def __init__(self, prefix=".") -> None:
        """Constructor for the mobile price dataset.

        Args:
            prefix (str, optional): Prefix for the path. Defaults to ".".
        """

        self.path = prefix + "/data/"

        self.__load()

    def __load(self) -> None:
        """Load the data from csv files.
        """

        self.description = pd.read_csv(self.path + "description.csv")
        self.df_train = pd.read_csv(self.path + "train.csv")
        self.df_test = pd.read_csv(self.path + "test.csv")
        self.df_train_fe = self.df_train.copy()
        self.df_test_fe = self.df_test.copy()

        self.__feature_engineering()

        self.dict_data = {
            # Description
            "description": self.description,

            # Train data
            "train": self.df_train,
            "train_x": self.df_train.drop(["price_range"], axis=1),
            "train_fe": self.df_train_fe,
            "train_fe_x": self.df_train_fe.drop(["price_range"], axis=1),

            # Train target
            "train_y": self.df_train["price_range"],

            # Test data
            "test": self.df_test,
            "test_x": self.df_test.drop(["id"], axis=1),
            "test_fe": self.df_test_fe,
            "test_fe_x": self.df_test_fe.drop(["id"], axis=1),
            
            # Test target
            "test_y": self.df_test["id"],          
        }

    def __feature_engineering(self) -> None:
        """Feature engineering. Create some handcrafted features.
        """

        self.df_train_fe = feature_engineering(self.df_train_fe)
        self.df_test_fe = feature_engineering(self.df_test_fe)

    def get_data(self) -> dict:
        return self.dict_data


def feature_engineering(df):
    """Feature engineering. Create some handcrafted features.

    Args:
        df (pandas DataFrame): DataFrame with the data

    Returns:
        df (pandas DataFrame): DataFrame with the extended data
    """
    
    # Camera
    # Total Camera Resolution: Combine front and primary camera resolutions
    df["cam_res"] = df["fc"] + df["pc"]

    # Front Camera-to-Back Camera Ratio: Calculate ratio of front to back cameras
    df["fc_pc_ratio"] = df["fc"] / df["pc"]

    # Screen
    # Screen Size: Combine screen height and width
    df["screen_size"] = df["sc_h"] * df["sc_w"]

    # Screen Aspect Ratio: Calculate screen aspect ratio
    df["screen_aspect_ratio"] = df["sc_h"] / df["sc_w"]

    # Pixel Per Inch: Calculate pixel per inch
    df["ppi"] = (df["px_height"] ** 2 + df["px_width"] ** 2) ** 0.5 / df["screen_size"]

    # Pixel Density: Calculate pixel density of the screen
    df["pixel_density"] = (df["px_height"] * df["px_width"]) / df["screen_size"]

    # CPU and Memory
    # Core-to-RAM Ratio (core_ram_ratio): Calculate ratio of cores to RAM
    df["core_ram_ratio"] = df["n_cores"] / df["ram"]

    # Processor Speed: Calculate processor speed
    df["processor_speed"] = df["clock_speed"] * df["n_cores"]

    # Core-to-Wieght Ratio: Calculate ratio of cores to weight
    df["core_weight_ratio"] = df["n_cores"] / df["mobile_wt"]

    # Memory-to-Weight Ratio: Calculate ratio of internal memory to weight
    df["mem_weight_ratio"] = df["int_memory"] / df["mobile_wt"]

    # Battery
    # Battery Efficiency: Calculate battery efficiency score
    df["battery_efficiency"] = df["battery_power"] / df["talk_time"]

    # Battery Screen Ratio: Calculate ratio of battery power to screen size
    df["battery_screen_ratio"] = df["battery_power"] / df["screen_size"]

    # Battery Weight Ratio: Calculate ratio of battery power to weight
    df["battery_weight_ratio"] = df["battery_power"] / df["mobile_wt"]

    # Other
    # Connectivity Score: Create a score for connectivity options
    df["connectivity_score"] = df["blue"] + df["three_g"] + df["four_g"] + df["wifi"]

    # Volume: Calculate volume of the phone
    df["volume"] = df["sc_h"] * df["sc_w"] * df["m_dep"]

    # Density: Calculate density of the phone
    df["density"] = df["mobile_wt"] / df["volume"]

    # Replace nan and inf values with 0.0
    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    return df


if __name__ == "__main__":
    
    MPD = MP_Dataset()

    dict_data = MPD.get_data()

    print(dict_data.keys())
    print(dict_data["train"].shape, dict_data["test"].shape)
    
