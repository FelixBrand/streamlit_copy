import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

FONT = dict(family="Arial", size=20,)
FONT_TITLE = dict(family="Arial", size=22)

# Dataframes for example-Heartbeats
df_kaggle = pd.read_pickle("data_selection\\Resampled_kaggle_data")
df_original = pd.read_pickle("data_selection\\Resampled_original_data")

def example_heartbeats(dataset_selection, colors):
    """Plots example-heartbeats"""
    class_index = [0, 162, 324, 486, 648]
    if dataset_selection == "Kaggle":
        time = np.arange(0, 187/125, 1/125)
        i = np.random.randint(0, 161)
        fig = go.Figure()
        for cat in [0, 1, 2, 3, 4]:
            fig.add_trace(go.Scatter(x=time, y=df_kaggle.iloc[class_index[cat] + i, 1:-1],
                                name=f"Class {cat}",line=dict(color=colors[cat])))
        fig.update_layout(title=dict(text='Random examples from Kaggle-Dataset', font=FONT_TITLE),
                          xaxis_title=dict(text='Time / s', font=FONT),
                          yaxis_title=dict(text='ECG-Values /  arb.unit', font=FONT),
                          legend_title=dict(text="Arrythmia", font=FONT),
                          template = "ggplot2",
                          font=FONT,
                          width=900,
                          height=500,)

    elif dataset_selection == "Original":
        time = np.arange(0, 310/180, 1/180)
        i = np.random.randint(0, 161)
        fig = make_subplots(rows=2, cols=1, vertical_spacing = 0.15,
                            subplot_titles=("Channel L2", "Channel V5"))
        for cat in [0, 1, 2, 3, 4]:
            fig.add_trace(go.Scatter(x=time, y=df_original.iloc[class_index[cat] + i, 0:315],
                                        name=f"Class {cat}", legendgroup = str(cat),
                                        line=dict(color=colors[cat])),
                            row=1, col=1)
            fig.add_trace(go.Scatter(x=time, y=df_original.iloc[class_index[cat] + i, 315:-1],
                                        name=f"Class {cat}", legendgroup = str(cat),
                                        line=dict(color=colors[cat])),
                            row=2, col=1)
        fig.update_layout(title=dict(text='Random examples from MIT-Dataset', font=FONT_TITLE),
                        xaxis2_title=dict(text='Time / s', font=FONT),
                        yaxis1_title=dict(text='ECG-Channel L2 / a.u.', font=FONT),
                        yaxis2_title=dict(text='ECG-Channel V5 / a.u.', font=FONT),
                        legend_title=dict(text="Arrythmia", font=FONT),
                        # legend = dict(font=dict(size=16)),
                        template = "ggplot2",
                        width=900,
                        height=700,)
        for i, yaxis in enumerate(fig.select_yaxes(), 1):
            legend_name = f"legend{i}"
            fig.update_layout({legend_name: dict(y=yaxis.domain[1], yanchor="top")},
                              showlegend=True)
            fig.update_traces(row=i, legend=legend_name)

    return fig


# %% Exploration
def class_distribution(colors):
    """ Function to plot Pie-chart with class distribution"""
    labels_long = {0: "Normal beat",
            1: "Supraventricular premature beat",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular and normal beat",
            4: "Unclassifiable beat"}

    distribution = pd.read_pickle("data\\MIT_classdistribution")
    custom_labels = [f'Class {i}: {labels_long[i]}' for i in distribution.index]
    fig = go.Figure(go.Pie(labels=custom_labels, values=distribution.values,
                           marker=dict(colors=colors)))
    fig.update_layout(title=dict(text='Class Distribution', font=FONT_TITLE),
                      template = "ggplot2",
                      width=1000,
                      height=500,
                      font = FONT,
                      legend = dict(font=FONT),
                      legend_title = dict(text="Classes", font=FONT))

    return fig


def plot_sigmoid_weights():
    """ Displays Sigmoid-weights"""

    def sigmoid_weights(weights):
        """ Custom modification of the sigmoid function to have a little bit of different values"""
        weights = np.array(weights)
        weights = weights / np.min(weights)
        weights =  1/(1+np.exp(-(weights/10-0.9)))
        weights_dict = {i: weights[i] for i in range(len(weights))}
        return weights_dict

    class_weights = {0: 0.23988963006158529,
                     1: 7.638768115942029,
                     2: 3.4197891321978915,
                     3: 29.694366197183097,
                     4: 2.670847189231987}
    sigmoid = sigmoid_weights(list(class_weights.values()))

    fig = go.Figure()
    x = np.linspace(np.min(list(class_weights.values())), np.max(list(class_weights.values())))
    fig.add_trace(go.Scatter(x=x, y=list(sigmoid_weights(x).values()),
                            showlegend=False,
                            line=dict(width=2, color='blue'),
                            hoverinfo='none'
                            ))
    fig.add_trace(go.Scatter(x=list(class_weights.values()),
                            y=list(sigmoid.values()),
                            hovertext=[ f"Class {i}" for i in list(class_weights.keys())],
                            marker=dict(
                                    symbol='x',  # Use 'x' as the marker symbol
                                    size=10,     # Size of the marker
                                    color='red', # Color of the marker
                                    line=dict(
                                        width=2,
                                        color='black'  # Border color of the marker
                                    )),
                            mode="markers",
                            showlegend=False,
                            name="Weights",
                            ))
    fig.update_layout(
        title=dict(text="Sample Weights", font=FONT_TITLE),
        xaxis_title=dict(text="Balanced Weights", font=FONT),
        yaxis_title=dict(text="Sigmoid Weights", font=FONT),
        template = "ggplot2",
        width=500,
        height=350,
    )
    return fig


def shap_plots(cat):
    """ Interpretability according to shap-values"""
    values1 = np.load("shapvalues_000_127.npy")
    values2 = np.load("shapvalues_128_256.npy")
    values3 = np.load("shapvalues_256_384.npy")
    shap_values =np.concatenate((values1, values2, values3))
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots

    mean_values = np.mean(shap_values, axis=0)
    std_upper = mean_values + np.std(shap_values, axis=0)
    std_lower = mean_values - np.std(shap_values, axis=0)
    time = np.arange(0, 315)

    fig = make_subplots(rows=2, cols=1,)# subplot_titles=("Time Series Plot 1", "Time Series Plot 2"))
    # L2-Channel
    fig.add_trace(go.Scatter(
            name='L2',
            x=time, y=mean_values[0:315, cat],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
            name='+ std',
            x=time, y=std_upper[0:315, cat],
            mode='lines', line=dict(width=0),
            marker=dict(color="#444"),
            showlegend=False
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
            name='- std',
            x=time, y=std_lower[0:315, cat],
            marker=dict(color="#444"),
            mode='lines', line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ), row=1, col=1)

    # V5-Channel
    fig.add_trace(go.Scatter(
            name='V5',
            x=time, y=mean_values[315:, cat],
            mode='lines',
            line=dict(color='rgb(200, 119, 180)'),
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
            name='+ std',
            x=time, y=std_upper[315:, cat],
            mode='lines', line=dict(width=0),
            marker=dict(color="#444"),
            showlegend=False
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
            name='- std',
            x=time, y=std_lower[315:, cat],
            marker=dict(color="#444"),
            mode='lines', line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ), row=2, col=1)
    FONT = dict(family="Arial", size=20,)
    FONT_TITLE = dict(family="Arial", size=22)

    fig.update_layout(
        yaxis1_title=dict(text='L2 / a.u.', font=FONT),
        yaxis2_title=dict(text='V5 / a.u.', font=FONT),
        xaxis2_title=dict(text='Indices', font=FONT),
        yaxis1=dict(range=[np.min(std_lower[:,cat]), np.max(std_upper[:,cat])]),
        yaxis2=dict(range=[np.min(std_lower[:,cat]), np.max(std_upper[:,cat])]),
        # yaxis_title=dict(text='Wind speed (m/s)', font=FONT),
        title=dict(text=f'Shap-Values - Class {cat}', font=FONT_TITLE),
        template = "ggplot2",
        width=1000,
        height=750,
        hovermode="x"
    )
    return fig




# %% Modelling - Boxplot

def mlflow_boxplot(mlflow_data, radio_parameter, radio_color, radio_dataset, radio_metric, radio_class):
    """ Function to plot the Boxplot of the mlflow-database according to the radio-Buttons. """
    # Dict to convert radio-text to mfllow-parameter-value
    convert_parameter = {"Model": "params.Model",
                         "Batch size": "params.batch_size",
                         "Class weights": "params.Weights",
                         "Data": "params.Data",
                         "Kernel size": "params.Kernel_size"}

    # For the weighted"-Metric
    classweights = np.array([87887, 2760, 6165, 710, 7894])
    classweights = classweights / np.sum(classweights)
    # Calculation of weighted and makro metrics
    for dataset in ["", "validation_"]:
        for metric in ["Precision", "Recall", "F1_score"]:
            temp_makro = mlflow_data["metrics." + dataset+metric+"_class1"]*0
            temp_weighted = mlflow_data["metrics." + dataset+metric+"_class1"]*0
            for classes in [0, 1, 2, 3, 4]:
                temp_makro = temp_makro + mlflow_data["metrics." + dataset +
                                                      metric+"_class" + str(classes)] / 5
                temp_weighted = temp_weighted + mlflow_data["metrics." + dataset +
                                                            metric+"_class" + str(classes)] \
                                                                * classweights[classes]
            mlflow_data["metrics." + dataset+metric+"_makro"] = temp_makro
            mlflow_data["metrics." + dataset+metric+"_weighted"] = temp_weighted

    # Get Column-name of the radio-buttons
    colname = "metrics."
    # Dataset-Radio
    if radio_dataset == "Validation":
        colname = colname + "validation_"  # for train nothing necessary
    # Metric-Radio
    if radio_metric == "Precision":
        colname = colname + "Precision_"
    elif radio_metric == "Recall":
        colname = colname + "Recall_"
    elif radio_metric == "F1":
        colname = colname + "F1_score_"
    elif radio_metric == "Loss":
        colname = colname + "loss"
    # Class-Metric
    if radio_metric != "Loss":
        if radio_class == "Makro":
            colname = colname + "makro"
        elif radio_class == "Weighted":
            colname = colname + "weighted"
        else:
            colname = colname + "class" + radio_class[-1]

    fig = px.box(mlflow_data,
                 x=convert_parameter[radio_parameter],
                 y=colname,
                 color=convert_parameter[radio_color],
                 points="all",
                 custom_data=['params.Model', 'params.Weights',
                             'params.Data', 'params.batch_size',
                             'params.Kernel_size', "index"],
                 title=radio_dataset + " " + radio_metric)
    fig.update_traces(hovertemplate="<br>".join([
                        radio_metric + ": %{y}",
                        "Model: %{customdata[0]}",
                        "Weights: %{customdata[1]}",
                        "Data: %{customdata[2]}",
                        "Batch Size: %{customdata[3]}",
                        "Kernel Size: %{customdata[4]}",
                        "Index: %{customdata[5]}"
                                                ]))

    fig.update_layout(title=radio_dataset + " " + radio_metric,
                        xaxis_title=dict(text=radio_parameter, font=FONT),
                        yaxis_title=dict(text=radio_metric, font=FONT),
                        legend_title=dict(text="Metrics", font=FONT),
                        template = "ggplot2",
                        width=1200,
                        height=600,
                        )
    fig.update_xaxes(tickangle=-35)

    return fig


# %% Modelling: Show training history
def import_training_history(index, mlflow_data):
    """ Loads training-history from mlflow-artifacts """
    index = int(index)
    path = "..\\..\\notebooks\\mlartifacts\\0\\" + mlflow_data.loc[index, "run_id"] + "\\artifacts"
    matching_file = [
        f for f in os.listdir(path)
        if f.startswith("Training_History") and f.endswith(".html")
                    ][0]
    with open(path + "\\" + matching_file, "r",) as f:
        plot_html = f.read()

    return plot_html

