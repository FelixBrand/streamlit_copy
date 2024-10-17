
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px



def print_crosstab(y_train_labels, y_pred_train, y_test_labels, y_pred_test, normalize=False):
    """ 
    Function to print the Crosstab between the prediction and the actual class for both the train and the test-data
    """
    test_crosstab = (pd.crosstab(y_test_labels, y_pred_test, rownames=["Real"], colnames=["Prediction Test"], normalize=normalize))
    train_crosstab = (pd.crosstab(y_train_labels, y_pred_train, rownames=["Real"], colnames=["Prediction Train"], normalize=normalize))
    if normalize is not False:
        test_crosstab = test_crosstab*100
        train_crosstab = train_crosstab*100
    empty_col = pd.DataFrame(test_crosstab.iloc[0])
    empty_col = empty_col * 0
    empty_col = empty_col.replace(0, " || ")
    empty_col.columns=["Sep"]

    crosstab = train_crosstab.join(empty_col, lsuffix="Train ")
    crosstab = crosstab.join(test_crosstab, rsuffix="Val ")
    return crosstab



def plot_metrics_from_history_express(history, title=None, show=True):
    labels = list(history.history.keys())
    epochs = len(history.history[labels[0]])
    colors = ["chocolate", "orange", "royalblue", "forestgreen", "orchid", "yellowgreen"]
    n_val, n = 0, 0
    fig = go.Figure()
    for idx in range(len(labels)//2):

        for idx_2 in [idx, idx+len(labels)//2]:
            metric = labels[idx_2]
            if "val" in metric:
                linedash = "solid"
                test = 1
                color = colors[np.mod(n_val, len(colors))]
                symbol = "x-dot"
                n_val += 1

            else:
                linedash = "dot"
                test = 1
                color = colors[np.mod(n, len(colors))]
                symbol = "circle"
                n += 1
            if "f1_score" in metric:
                # f1 has one score for each class
                for i_f1score in range(5):
                    fig.add_trace(go.Scatter(x=np.arange(1, epochs+1), y=np.array(history.history[metric])[:,i_f1score], 
                                             name=f"{metric} Class {i_f1score}", hovertext=f"{metric} Class {i_f1score}", 
                                             marker_color=color, marker_size=10, marker_symbol=str(i_f1score), marker_line_width=1,
                                             line_dash=linedash))
            else:
                fig.add_trace(go.Scatter(x=np.arange(1, epochs+1), y=np.array(history.history[metric]), 
                                        name=metric, hovertext=metric, 
                                        marker=dict(symbol=symbol, line_width=1, color=color, size=10),
                                        # marker_color=color, marker_size=100, marker_symbol=symbol, marker_line_width=115,
                                        line_dash=linedash))

    fig.update_layout(title=title,
                      xaxis_title="Epochs", 
                      yaxis_title="Metrics", 
                      legend_title="Metrics",
                      template = "ggplot2",
                      width=1500,
                      height=750,
                      )
    if show == True:
        fig.show()
    
    
    return fig




