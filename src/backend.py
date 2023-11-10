#---------------------------------------------------------------------------------------------------#
# File name: backend.py                                                                             #
# Autor: Chrissi2802                                                                                #
# Created on: 07.11.2023                                                                            #
# Content: This file provides the backend.                                                          #
#---------------------------------------------------------------------------------------------------#


import pandas as pd
import yaml
from pydantic import BaseModel, conlist
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from data import MP_Dataset
from prediction import MP_Prediction


# Information
with open("info.yaml", "r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

dict_params_user = yaml_data["dict_params_user"]


# Data
MPD = MP_Dataset(prefix=dict_params_user["prefix"])
dict_data = MPD.get_data()
columns = dict_data["train_x"].columns

class Input_data(BaseModel):
    data: List[conlist(float, min_items=20, max_items=20)]


# Prediction
MPP = MP_Prediction(dict_params_user)


# App
app = FastAPI(
    title="Mobile Price Classification API", 
    description="API for the Mobile Price Classification", 
    version="0.1",
)

# Configuration for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Here you can specify the permitted origins (e.g "http://localhost", "https://example.com")
    allow_credentials=True,
    allow_methods=["POST"],     # Permitted HTTP methods
    allow_headers=["Authorization"],    # Permitted HTTP headers
)


@app.post("/predict", tags=["predictions"])
async def get_prediction(data: Input_data):
    """Get the prediction for the price range as post request.

    Args:
        data (Input_data): Input data / features

    Returns:
        {} (dict): Predicted price range and the probability for each class
    """

    data = pd.DataFrame(data.data, columns=columns)
    prediction = MPP.predict(data)

    # Round the probabilities
    prediction["pred_proba"] = prediction["pred_proba"].round(3)

    return {
        "pred": prediction["pred"][0].tolist(),
        "pred_proba": prediction["pred_proba"][0].tolist(),
    }


if __name__ == "__main__":

    import uvicorn

    # Start the FastAPI application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # http://localhost:8000/docs

