# %%
# imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import platform


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
# create an OpenCV image visualizing xy coordinates as circles

def create_image(
        xy_coords: np.ndarray,
        point_size: int = 10) -> np.ndarray:
    img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), np.uint8)

    for x, y in xy_coords:
        cv2.circle(img, (int(np.round(x)), int(np.round(y))), point_size, (255, 255, 255), -1)

    return img

# %%
# %%
# function: view

def view(matrix: np.ndarray, scale: float = 1.0, text: str = None, swap_rb: bool = True) -> None:
    """
    Quickly view a matrix.

    Args:
        matrix (np.ndarray): The matrix to view.
        scale (float, optional): Scale the matrix (image) axes. 0.5 corresponds to halving the width and height, 2 corresponds to a 2x zoom. Defaults to 1.
        text (str, optional): The text to display at the bottom left corner as an overlay.
        swap_rb (bool, optional): Whether to swap red and blue channels before displaying the matrix. Defaults to True.
    """
    # opencv needs bgr order, so swap it if input matrix is colored
    to_show = matrix.copy()
    if swap_rb and len(to_show.shape) > 2:
        to_show[:, :, [0, -1]] = to_show[:, :, [-1, 0]]
    if scale != 1:
        h, w = to_show.shape[:2]
        h_scaled, w_scaled = [int(np.ceil(ax * scale)) for ax in [h, w]]
        to_show = cv2.resize(to_show, (h_scaled, w_scaled))
    if text != None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            to_show, text, (12, to_show.shape[1] - 12), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # avoid hanging windows on Mac OS
    if platform.system() == "Darwin":
        cv2.startWindowThread()
    cv2.imshow("view", to_show.astype(np.uint8))
    # avoid unmovable windows on Mac OS
    if platform.system() == "Darwin":
        cv2.moveWindow("view", 50, 50)
    cv2.waitKey(0)
    # avoid hanging windows on Mac OS
    if platform.system() == "Darwin":
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# %%
view(create_image(xy_coords), scale=0.25)
# %%
text_xy_coords = np.array([[50, 500]])
view(create_image(text_xy_coords, point_size=10), scale=0.5, text="hello")
# %%

view(create_image(get_xy_coords(filter_frame(df, "N16", visiting_point=1, frame=1)), point_size=10), scale=0.25)
# %%
