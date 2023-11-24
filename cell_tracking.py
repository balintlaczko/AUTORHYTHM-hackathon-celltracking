# %%
# imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# %%
# constants
IMG_WIDTH = 2048
IMG_HEIGHT = 2048

# %%
# read csv
csv_path = "tabular-data/hackathon_dataset_S2cells.csv"
df = pd.read_csv(csv_path)
df.head()



# %%
df.tail()

# %%
# helper function to retrieve coords in a frame
# params: well id, visiting point, frame number

def filter_frame(
        df: pd.DataFrame, 
        well_id: str, 
        visiting_point: int, 
        frame: int
        ) -> pd.DataFrame:
    # filter for well id
    df_filtered = df[df["Image_Metadata_Well"] == well_id]

    # filter for visiting point
    df_filtered = df_filtered[df_filtered["Image_Metadata_Multipoint"] == visiting_point]

    # filter for frame
    df_filtered = df_filtered[df_filtered["timepoint"] == frame]

    return df_filtered

# %%
# test filter_frame

df_filtered = filter_frame(df, "N16", visiting_point=1, frame=1)
df_filtered.head()

# %%
# helper func to get an array of xy coords from a df

def get_xy_coords(df: pd.DataFrame) -> np.ndarray:
    xy_coords = df[["Cells_Location_Center_X", "Cells_Location_Center_Y"]].to_numpy()
    return xy_coords

# %%
# test get_xy_coords

xy_coords = get_xy_coords(df_filtered)
xy_coords.shape # (270, 2)


# %%
