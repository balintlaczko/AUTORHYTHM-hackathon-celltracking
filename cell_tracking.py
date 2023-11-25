# %%
# imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import platform
from scipy.spatial import KDTree


# %%
# constants
IMG_WIDTH = 2048
IMG_HEIGHT = 2048

# %%
# read csv
csv_path = "tabular-data/hackathon_dataset_S2cells_231124.csv"
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
# extract coordinates from two frames
frame_0_points = get_xy_coords(filter_frame(df, "N16", visiting_point=1, frame=2))
frame_0_points = frame_0_points.astype(np.float32)
frame_1_points = get_xy_coords(filter_frame(df, "N16", visiting_point=1, frame=3))
frame_1_points = frame_1_points.astype(np.float32)

# generate images
# previous frame
frame_0 = create_image(frame_0_points, point_size=10)
# convert to grayscale
frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

# current frame
frame_1 = create_image(frame_1_points, point_size=10)
# convert to grayscale
frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

# Parameters for lucas kanade optical flow 
# lk_params = dict( winSize = (1000, 1000), 
#                   maxLevel = 4, 
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
#                               1000, 0.01))
lk_params = dict( winSize = (100, 100), 
                  maxLevel = 4, 
                  criteria = (cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# %%

# calculate optical flow 
p1, st, err = cv2.calcOpticalFlowPyrLK(frame_0, frame_1, frame_0_points, None, **lk_params)


# %%
input_first_point = frame_0_points[15]
output_first_point = p1[15]
input_first_point, output_first_point, frame_1_points[1]

# %%
# fit a kd-tree to the input points and find the closest point to the output point
tree = KDTree(frame_0_points)
distance, index = tree.query(frame_1_points, p=2)
distance, index

# %%
# Select good points 
good_new = p1[st[:, 0] == 1] 
good_old = frame_0_points

# Create a mask image for drawing purposes 
mask = np.zeros(frame_0.shape + (3, )).astype(np.uint8) 
mask.shape

# reorder frame_1_points based on kdtree distance
frame_1_points_reordered = frame_1_points[index]

# draw the tracks 
for i, (new, old) in enumerate(zip(good_new,  
                                    good_old)): 
    a, b = new.ravel() 
    c, d = old.ravel() 
    a, b, c, d = int(a), int(b), int(c), int(d)
    mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 20) 

frame_0_bgr = cv2.cvtColor(frame_0, cv2.COLOR_GRAY2BGR)
frame_1_bgr = cv2.cvtColor(frame_1, cv2.COLOR_GRAY2BGR)
img = cv2.add(frame_1_bgr, mask) 

# view(img, scale=0.25)
# %%
# save frame 0 and frame 1
cv2.imwrite("frame_0.png", frame_0)
cv2.imwrite("frame_1.png", frame_1)
cv2.imwrite("frame_1_tracking.png", img)


# %%
# kdtree
frames = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a mask image for drawing purposes 
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.uint8)

# global var for persistent ids
ids2xy = {}


for frame_id, frame in enumerate(frames):
    print(frame_id, frame)
    # extract coordinates from two frames
    frame_0_points = get_xy_coords(filter_frame(df, "N15", visiting_point=1, frame=frame))
    # frame_0_points = frame_0_points.astype(np.float32)
    frame_1_points = get_xy_coords(filter_frame(df, "N15", visiting_point=1, frame=frame+1))
    # frame_1_points = frame_1_points.astype(np.float32)

    # generate images
    # previous frame
    frame_0 = create_image(frame_0_points, point_size=15)
    # current frame
    frame_1 = create_image(frame_1_points, point_size=15)

    # initialize persistent ids upon first frame
    if frame_id == 0:
        for id, xy in enumerate(frame_0_points):
            ids2xy[id] = xy

    # fit a kd-tree to the input points and find the closest point to the output point
    tree = KDTree(frame_0_points)
    # distance, index = tree.query(frame_1_points, p=2, distance_upper_bound=10000)
    distance, index = tree.query(frame_1_points, p=2, distance_upper_bound=50)

    # Create a mask image for writing label numbers 
    labels_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.uint8)

    for index_id, prev_index in enumerate(index):
        # if it was outside the distance upper bound, skip
        if prev_index == len(frame_0_points):
            continue
        
        # if you get a valid point, take the xy for prev_index
        # and find the identical xy in ids2xy
        # then draw a line from that point to the current point
        prev_xy = frame_0_points[prev_index]
        prev_id = None
        for id, xy in ids2xy.items():
            if np.array_equal(xy, prev_xy):
                prev_id = id
                # update ids2xy at label
                ids2xy[id] = frame_1_points[index_id]
                # write label number at xy on labels_mask
                new_xy = frame_1_points[index_id]
                labels_mask = cv2.putText(labels_mask, str(id), (int(new_xy[0])+20, int(new_xy[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                break

        new_point = frame_1_points[index_id]
        a, b = new_point
        old_point = frame_0_points[prev_index]
        c, d = old_point
        a, b, c, d = int(a), int(b), int(c), int(d)
        mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 15)


    img = cv2.add(frame_1, mask)
    # add labels_mask to img
    img = cv2.add(img, labels_mask)

    # save images
    if frame_id == 0:
        cv2.imwrite(f"frame_{frame}.png", frame_0)
        cv2.imwrite(f"frame_{frame+1}_tracking.png", img)
    else:
        cv2.imwrite(f"frame_{frame+1}_tracking.png", img)

# %%
# collect into function

def render_tracking_visualization(
        df: pd.DataFrame,
        well_id: str,
        visiting_point: int,
        frames: list,
        distance_upper_bound: int = 50,
        point_size: int = 15,
        path_color: tuple = (0, 0, 255),
        label_color: tuple = (255, 0, 0),
        destination: str = "."
        ) -> None:
    # Create a mask image for drawing purposes 
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.uint8)

    # global var for persistent ids
    ids2xy = {}

    # create destination folder if it doesn't exist
    os.makedirs(destination, exist_ok=True)


    for frame_id, frame in enumerate(frames):
        # extract coordinates from two frames
        frame_0_points = get_xy_coords(filter_frame(df, well_id=well_id, visiting_point=visiting_point, frame=frame))
        # frame_0_points = frame_0_points.astype(np.float32)
        frame_1_points = get_xy_coords(filter_frame(df, well_id=well_id, visiting_point=visiting_point, frame=frame+1))
        # frame_1_points = frame_1_points.astype(np.float32)

        # generate images
        # previous frame
        frame_0 = create_image(frame_0_points, point_size=point_size)
        # current frame
        frame_1 = create_image(frame_1_points, point_size=point_size)

        # initialize persistent ids upon first frame
        if frame_id == 0:
            for id, xy in enumerate(frame_0_points):
                ids2xy[id] = xy

        # fit a kd-tree to the input points and find the closest point to the output point
        tree = KDTree(frame_0_points)
        distance, index = tree.query(frame_1_points, p=2, distance_upper_bound=distance_upper_bound)

        # Create a mask image for writing label numbers 
        labels_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.uint8)

        for index_id, prev_index in enumerate(index):
            # if it was outside the distance upper bound, skip
            if prev_index == len(frame_0_points):
                continue
            
            # if you get a valid point, take the xy for prev_index
            # and find the identical xy in ids2xy
            # then draw a line from that point to the current point
            prev_xy = frame_0_points[prev_index]
            prev_id = None
            for id, xy in ids2xy.items():
                if np.array_equal(xy, prev_xy):
                    prev_id = id
                    # update ids2xy at label
                    ids2xy[id] = frame_1_points[index_id]
                    # write label number at xy on labels_mask
                    new_xy = frame_1_points[index_id]
                    labels_mask = cv2.putText(labels_mask, str(id), (int(new_xy[0])+20, int(new_xy[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)
                    break

            # unpack new point
            a, b = frame_1_points[index_id]
            # unpack old point
            c, d = frame_0_points[prev_index]
            # round and convert to int
            a, b, c, d = int(np.round(a)), int(np.round(b)), int(np.round(c)), int(np.round(d))
            mask = cv2.line(mask, (a, b), (c, d), path_color, 15)

        # add mask to frame_1
        img = cv2.add(frame_1, mask)
        # add labels_mask to img
        img = cv2.add(img, labels_mask)

        # save images
        # upon first frame, save frame_0 as well
        if frame_id == 0:
            cv2.imwrite(os.path.join(destination, f"frame_{frame}.png"), frame_0)
        # and always save the tracking for the next frame
        cv2.imwrite(os.path.join(destination, f"frame_{frame+1}_tracking.png"), img)

# %%
# generate tracking visualizations for all wells and visiting points

# get all unique well ids
well_ids = df["Image_Metadata_Well"].unique()

# root folder for tracking visualizations
root = "tracking_visualizations"

for well_id in well_ids:
    print(well_id)
    # get all unique visiting points for this well
    visiting_points = df[df["Image_Metadata_Well"] == well_id]["Image_Metadata_Multipoint"].unique()
    for visiting_point in visiting_points:
        print(visiting_point)
        # get all unique frames for this well and visiting point
        frames = df[(df["Image_Metadata_Well"] == well_id) & (df["Image_Metadata_Multipoint"] == visiting_point)]["timepoint"].unique()
        # sort frames
        frames = sorted(frames)
        # remove first (bst) and last frame
        frames = frames[1:-1]
        print(frames)
        # generate tracking visualizations
        render_tracking_visualization(df, well_id, visiting_point, frames, destination=os.path.join(root, well_id, str(visiting_point)))
# %%
