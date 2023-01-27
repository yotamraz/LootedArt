import io
import os
import base64
import ast
import random
import traceback

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageStat
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def rand_color_dict(keys):
    color_dict = {}
    r = lambda: random.randint(0, 255)
    for key in keys:
        color = '#%02X%02X%02X' % (r(), r(), r())
        color_dict[key] = color
    return color_dict

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def load_images(df):
    images = []
    image_names = []
    embedded_rep = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        im = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)

        images.append(im)
        image_names.append(row["image_path"].split("/")[-1].replace(".jpg", ""))
        embedded_rep.append(ast.literal_eval(row["embedding"])[0])

    return images, image_names, embedded_rep


def resize_images(image):
    return cv2.resize(image, (int(image.shape[1] * 3), int(image.shape[0] * 3)), interpolation=cv2.INTER_LINEAR)


def plot_bar_plot(y_true, y_pred, y_scores):
    palette = {"true positive": color_map_group["11"],
               "true negative": color_map_group["00"],
               "false positive": color_map_group["01"],
               "false negative": color_map_group["10"]}

    df = pd.DataFrame(data=list(zip(y_true, y_pred, y_scores)), columns=["y_true", "y_pred", "y_scores"])
    df["error_type"] = error_type(df)
    df["score_range"] = df["y_scores"].apply(lambda x: float("%.1f" % x))
    df["score_group"] = df["y_scores"].apply(lambda x: 1 if x > .85 else 0)
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    sns.countplot(data=df, x="score_range", hue="error_type",
                  palette=palette)
    title = "Nozzle classification error as a function of confidence score"
    plt.title(title)
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    sns.countplot(data=df, x="score_group", hue="error_type",
                  palette=palette)
    title = "Nozzle classification error above/below 0.85% confidence"
    plt.title(title)
    plt.show()


def error_type(df):
    error_type_list = []
    for idx, row in df.iterrows():
        if row["y_true"] == 1 and row["y_pred"] == 1:
            error_type_list.append("true positive")
        elif row["y_true"] == 0 and row["y_pred"] == 0:
            error_type_list.append("true negative")
        elif row["y_true"] == 0 and row["y_pred"] == 1:
            error_type_list.append("false positive")
        else:
            error_type_list.append("false negative")
    return error_type_list


def calc_cosine_sim(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def calc_min_max_similarity(x1, x2):
    global_min = 1
    global_max = -1
    for i, point in enumerate(x1):
        sim = calc_cosine_sim(point, x2)
        if sim < global_min:
            global_min = sim
        elif sim > global_max:
            global_max = sim
    return global_min, global_max


def calc_conf_score(embedded_rep, true_labels, predicted_labels, method="knn"):
    """
    1. for every row do:
    2. find knn in the 128-dim space (fixed k for the meantime)
    3. use the confidence metric defined in (1) "Distance-based Confidence Score for Neural Network Classifiers - Amit Mandelbaum & Daphna Weinshell"
    4. add as a new column & return df
    """


    if method == "knn":
        # params
        k = 50
        w = 'distance'
        algo = 'auto'

        # aggregator
        scores = []

        # find neighbors
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(embedded_rep)

        # loop on samples
        for i, point in enumerate(embedded_rep):
            distances, indices = neigh.kneighbors([point])
            # remove reflexive point
            distances, indices = distances[:, 1:], indices[:, 1:]
            curr_labels = np.asarray([true_labels[j] for j in indices[0, :]]).reshape(1, -1)  # skip the first score
            numerator = np.sum(np.exp(-distances[curr_labels == predicted_labels[i]])) if len(curr_labels) > 0 else 0
            denominator = np.sum(np.exp(-distances))
            conf_score = numerator / denominator
            scores.append(conf_score)
        return scores

    elif method == "centroid":
        # find good nozzle centroid
        good_subset_X, good_subset_true, good_subset_predicted = embedded_rep[
                                                                 [j for j, _ in enumerate(true_labels) if _ == 1], :], \
                                                                 true_labels[true_labels == 1], predicted_labels[
                                                                     predicted_labels == 1]
        good_centroid = good_subset_X.mean(axis=0)
        # find bad nozzle centroid
        bad_subset_X, bad_subset_true, bad_subset_predicted = embedded_rep[
                                                              [j for j, _ in enumerate(true_labels) if _ == 0], :], \
                                                              true_labels[true_labels == 0], predicted_labels[
                                                                  predicted_labels == 0]
        bad_centroid = bad_subset_X.mean(axis=0)

        good_similarity_min, good_similarity_max = calc_min_max_similarity(good_subset_X, good_centroid)
        bad_similarity_min, bad_similarity_max = calc_min_max_similarity(bad_subset_X, bad_centroid)
        scores = []
        for i, (X, y) in enumerate(zip(embedded_rep, predicted_labels)):
            good_similarity = calc_cosine_sim(X, good_centroid)
            bad_similarity = calc_cosine_sim(X, bad_centroid)
            score = y * ((good_similarity - good_similarity_min) / (good_similarity_max - good_similarity_min)) + (
                        1 - y) * ((bad_similarity - bad_similarity_min) / (bad_similarity_max - bad_similarity_min))
            scores.append(score)

        return scores


app = Dash(__name__)

@app.callback(
    Output('graph-5', 'figure'),
    Input('dropdown1', 'value')
)
def update_by_color(value):

    fig = go.Figure(data=[go.Scatter(
        x=map[:, 0],
        y=map[:, 1],
        mode='markers',
        marker=dict(
            size=5,
            color=(255,30,20),
        )
    )])

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        showlegend=True,
    )

    return fig


@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "96px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("image name " + str(image_names[num])),
        ])
    ]

    return True, bbox, children


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    df_path = "../ResNetSim/results/embedding.csv"

    print("########## Starting embedding space visualization server ##########")
    # read df
    df = pd.read_csv(df_path)
    # sample = df.sample(n=1000)

    print("Reading images from disk...")
    # Load the data
    images, image_names, embedded_rep = load_images(df)

    print("Computing t-sne mapping...")
    # t-SNE Outputs a 3 dimensional point for each image
    map = TSNE(
        random_state=0,
        n_components=2,
        verbose=0,
        perplexity=10,
        n_iter=5000).fit_transform(embedded_rep)

    # map = PCA(
    #     n_components=2, random_state=22
    # ).fit_transform(embedded_rep)

    #
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Dropdown(['Subset', 'Print Name', 'Confusion Matrix Group'], 'Subset', id='dropdown1'),
            dcc.Graph(id="graph-5", figure=None, clear_on_unhover=True,
                      style={'width': '100vw', 'height': '100vh', 'backgroundColor': 'black'}),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    app.run_server(debug=True)