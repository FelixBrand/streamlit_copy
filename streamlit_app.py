# %% Libary Imports:

# run with: streamlit run streamlit_app.py

import numpy as np
import pandas as pd
import streamlit as st
import os
import sys

sys.path.append("reports/Streamlit")
import plotly_graphs
import livedemo

# Paths in Streamlit-hosted are relative to main-folder, local hosted streamlit are relative to streamlit_app.py file
# Just put streamlit_app.py in main-folder!
if os.path.dirname(__file__)[0:2] == "C:":
    # local PC - paths are relativ to the "streamlit_app.py" file
    path = ""
else:
    # Streamlit-Host - paths are relative to git-main folder
    path = ""


# %% Data Import
labels_long = {0: "Normal beat",
            1: "Supraventricular premature beat",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular and normal beat",
            4: "Unclassifiable beat"}
# Color for all plotly-graphs (first -> Class 0, second: class 1, ...)
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Import ML-Flow-Database and minor modifications

mlflow_data = pd.read_pickle("reports/Streamlit/data/mlflow_database.pkl")
print(os.path.dirname(__file__))

mlflow_data.replace({"Combined_cnn_1": "Combined_CNN_1",
            "Convolution_smallkernel": "CNN_smallkernel",
            "Convolution_bigkernel": "CNN_bigkernel"}, inplace=True)
mlflow_data["params.batch_size"] = mlflow_data["params.batch_size"].astype(int)
mlflow_data["params.Kernel_size"] = mlflow_data["params.Kernel_size"].fillna("None")
mlflow_data = mlflow_data.reset_index()


# %% Streamlit Pages + Sidebar

# Set of Side-width
css='''
<style>
    section.main > div {max-width:100rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

st.title("Heartbeat Arrhythmia Detection")
pages = ["Introduction", "Available Data", "Data Exploration",
          "Modelling", "Modelling Results", "Live Test", "Interpretability", "Conclusion", "About"]
st.sidebar.title("Table of contents")
page = st.sidebar.radio("Go to", pages)
st.sidebar.write("# ")
st.sidebar.write("# ")
st.sidebar.write("# ")
st.sidebar.write("## Participants: \nFelix Brand")
st.sidebar.write("# ")
st.sidebar.write("# ")
st.sidebar.write("# ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write("## Class Legend")
st.sidebar.markdown("""
                    - 0: Normal beat
                    - 1: Supraventricular premature beat
                    - 2: Premature ventricular contraction
                    - 3: Fusion of ventricular and normal beat
                    - 4: Unclassifiable beat
                    """)

# %% Page "Introduction"

if page == "Introduction":  # Match with "pages"-entries
    st.write(f"## {page}")

    st.write("#### PQRST heartbeat scheme")
    col1, col2 = st.columns(2)
    col1.markdown("""
        The PQRST scheme is a method used of analysing and interpreting the electrocardiogram (ECG or EKG) waveforms, \
                   which represent the electrical activity of the heart. Each letter in 'PQRST' corresponds to specific components of the cardiac cycle:

        - **P wave:** Represents atrial depolarisation. This is the electrical impulse that causes the atria to contract and pump blood into the ventricles.
        - **QRS complex:** This complex reflects the main pumping action of the heart and represents the ventricular depolarisation. This complex is a combination of three waves:
          - **Q wave:** The initial negative deflection.
          - **R wave:** The subsequent positive deflection.
          - **S wave:** The negative wave that follows the R wave.
        - **T wave:** Represents ventricular repolarisation. This is when the ventricles recover and prepare for the next heartbeat.
                  """)
    col2.image("reports/Streamlit/figures/Heartbeat_scheme.png",
             caption="https://www.analyticsvidhya.com/blog/2021/07/artificial-neural-network-simplified-with-1-d-ecg-biomedical-data/")

    st.divider()

    st.write("#### Arrhythmia")
    st.markdown("""
    A heartbeat arrhythmia, also known simply as an arrhythmia, refers to an irregular heartbeat. \n
    Some heart arrhythmias are harmless, but they can indicate serious disease such as coronary artery disease, heart failure, heart valve disorders, electrolyte imbalances and many others. \n
    Heartbeats are classified as follows:
    - 0: Normal beat
    - 1: Supraventricular premature beat
    - 2: Premature ventricular contraction
    - 3: Fusion of ventricular and normal beat
    - 4: Unclassifiable beat
                """)



# %% Page "Available Data"
if page == "Available Data":  # Match with "pages"-entries
    st.write(f"## {page}")


    # Overview of the Measurement-Setup:
    measurement_setup = {
        "Measurement type": "2-channel ECG á 30 min",
        "Patients": "47 patients, 48 records",
        "Classification": "2 independent cardiologists, annotations at R-Peak",
        "Additional information": "Age, Medication, Sex",
        "Sampling frequency": "360 Hz",
        "Frequency filter": "0.1 ... 100 Hz",
        "Measurement resolution": "11 bit",
        "Input voltage": "+- 10 mV",
        "Device": "Del-Mar Avionics model 445",
                        }
    st.write("### Measurement-Setup")
    measurement_setup = pd.DataFrame({"Setting": measurement_setup.keys(), "Value": measurement_setup.values()})
    # st.dataframe(measurement_setup, width=750) #, hide_index=True
    st.markdown(measurement_setup.style.hide(axis="index").to_html(), unsafe_allow_html=True)

    st.divider()
    st.markdown("""
            Two datasets can be used:
            - Kaggle preprocessed data
            - Original unprocessed data from MIT
            """)
    dataset_selection = st.selectbox("Dataset:", ("Kaggle", "Original"))

    col1, col2 = st.columns(2)
    shuffle = col2.button("Plot and Shuffle")
    if shuffle:

        fig = plotly_graphs.example_heartbeats(dataset_selection, colors)
        st.plotly_chart(fig)

    st.divider()

    st.write("### Differences in preprocessing:")
    _, center, _,  = st.columns(3)
    colname1, colname2, colname3 = ["Setting", "Kaggle", "Original"]
    dataset_comparision = pd.DataFrame([
        {colname1: "Sampling-Rate", colname2: "125 Hz - interpolation", colname3: "180 - undersampling"},
        {colname1: "Normalisation", colname2: "Min-Max-Scaling per Heartbeat", colname3: "Max-Scaling per patient"},
        {colname1: "Data", colname2: "One channel / unknown", colname3: "Both channels"},
        {colname1: "Beat Seperation", colname2: "R to R/T", colname3: "P to T"},
        {colname1: "...", colname2: "...", colname3: "..."},
                                        ])
    # st.dataframe(dataset_comparision, width=750)#, hide_index=True
    # st.table(dataset_comparision, )
    # st.dataframe(dataset_comparision.style.hide(axis="index"))
    st.markdown(dataset_comparision.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    st.divider()

    st.write("## Conclusion:")
    st.markdown("""
      In the following, the imported original MIT data is used:
      - Precise knowledge of data processing
      - No interpolation

      - Normalisation based on one patient (max-scaling)
      - PQRST scheme is reproduced
      - QRS complex (center point) is labelled with classification labels
    Additional information (age, sex, medications, device) is not used, as a patient classification would be unambiguous.
                """)

# %% Page Data Exploration
if page == "Data Exploration":
    st.write(f"## {page}")
    st.write("### Amount of data:")
    st.markdown("""
                - 105416 separate heartbeats
                - 315 samples per channel
                - sampling rate: 180 Hz
                - max. 1.75 s per heartbeat, filled with 0s
                - 2 channels (L2, V5)
                - ~ 55% 0-values
                """)

    # Pie-Chart of Class distribution
    fig = plotly_graphs.class_distribution(colors)
    center, _, = st.columns(2)
    center.plotly_chart(fig)


# %% Page "Modelling"
if page == "Modelling":
    st.write(f"## {page}")
    st.write("### Network architecture:")
    choice = ["Dense 1", "Dense 2", "CNN 1 - small kernel", "CNN 2 - big kernel",
              "Fusion Dense 1", "Fusion Dense 2", "Fusion CNN"]
    net_names = ["Dense_1", "Dense_2",
                 "Convolution_smallkernel", "Convolution_bigkernel",
                 "Combined_Dense_1", "Combined_Dense_2", "Combined_cnn_1"]
    selection = st.selectbox('Model', choice)
    index = choice.index(selection)
    _, center, _ = st.columns(3)
    center.image("reports/Streamlit/figures/model_figures/" + net_names[index] + ".jpg")

    st.divider()
    st.write("### Preprocessing:")
    st.write("Preprocessing like FFT, remove-offset and digital filter didn't increased the performance. ")

    st.divider()
    col1, col2 = st.columns(2)
    col1.write("### Weights:")
    col1.markdown("""
                Three different types of sample weights were used:
                - No weights: Each class has the same weight
                - Balanced weights: Each class has a weight that is inversely proportional to its occurrence.
                - Sigmoid weights: Compared to balanced weights, the weights of the underrepresented classes are reduced by a sigmoid function.""")
    col2.plotly_chart(plotly_graphs.plot_sigmoid_weights())

# %% Page "Modelling Results"
if page == "Modelling Results":
    st.write(f"## {page}")
    st.write("Hyperparameter: \n "
              "- Model \n"
              "- Batch size: 256, 512, 1024, 2048 \n"
              "- Class weights normal, balanced, sigmoid \n"
              "- Kernel size: 3, 5, 7\n"
              "- Data: L2, V5, [L2, V5], [L2 | V5]")
    st.write("### Boxplots for different kinds of hyperparameter")
    col1, col2, col3, col4, col5 = st.columns(5)
    radio_parameter = col1.radio("Parameter:", ("Model", "Batch size", "Class weights",
                                                 "Data", "Kernel size"))
    radio_color = col2.radio("Color-Parameter:", ("Model", "Batch size", "Class weights",
                                                  "Data", "Kernel size"))
    radio_dataset = col3.radio("Dataset:", ("Train", "Validation"))
    radio_metric = col4.radio("Metric:", ("Precision", "Recall", "F1", "Loss"))
    if radio_metric == "Loss":
        radio_class = col5.radio("Class:", ("Makro", "Weighted", "Class 0",
                                            "Class 1", "Class 2", "Class 3",
                                            "Class 4"), disabled=True)
    else:
        radio_class = col5.radio("Class:", ("Makro", "Weighted", "Class 0",
                                            "Class 1", "Class 2", "Class 3",
                                            "Class 4"))

    fig = plotly_graphs.mlflow_boxplot(mlflow_data, radio_parameter, radio_color,
                                       radio_dataset, radio_metric, radio_class)
    st.plotly_chart(fig)

    show_content = st.checkbox("Show training history:")

    if show_content:
        index = st.number_input("# Training History for index:",
                                    min_value=int(0), max_value=int(mlflow_data["index"].max()), value=0,
                                    step=int(1))
        if index is not None:
            plot_html = plotly_graphs.import_training_history(index, mlflow_data)

            st.components.v1.html(plot_html, height=800)

    st.divider()

    st.write("## Conclusion:")
    st.markdown("""Best results for:
                - Model: Combined_CNN_1
                - Batch size: 256
                - Data: L2 + V5
                - Kernel size: """)

    st.write("Balance Report for that combination and the crosstab")

# %% Page: "Live Test"
if page == "Live Test":
    st.write(f"## {page}")

    live = livedemo.livedemo()

    record_number = st.selectbox("Select a record:", live.patients)
    fig = live.record_plot(record_number)
    # fig.update_layout(dragmode="drawrect")
    area = st.plotly_chart(fig, on_select="rerun", selection_mode="box")

    st.json(area)

    # # Area is the selection in the plotly-Graph
    # if area["selection"]["box"] != []:
    #     area = np.mean(area["selection"]["box"][0]["x"])

    #     fig, l2_sep, v5_sep, real_class = live.single_heartbeat_plot(record_number, area)
    #     col1, col2 = st.columns(2, vertical_alignment="center")
    #     col1.plotly_chart(fig)

    #     text1, text2, text3 = live.prediction(l2_sep, v5_sep, real_class)
    #     col2.write(text1)
    #     col2.markdown(text2, unsafe_allow_html=True)
    #     col2.markdown(text3, unsafe_allow_html=True)


if page == "Interpretability":
    st.write(f"## {page}")
    classdict = {"Class 0": 0, "Class 1": 1, "Class 2": 2, "Class 3": 3, "Class 4": 4}
    radio_class = st.radio("Shape-Values of", classdict.keys(), horizontal=True)

    st.plotly_chart(plotly_graphs.shap_plots(classdict[radio_class]))




# %% Page "Conclusion"
if page == "Conclusion":
    st.write(f"## {page}")
    st.write("Here comes some text for the conclusion")

# %% Page "About"
if page == "About":
    st.write(f"## {page}")
    st.write("Participants:\n"
             "- Felix Brand\n"
             "\n"
             "Supervisor:\n"
             "- Francesco")
    st.write("## Data:")
    st.write("Original MIT Source:")
    st.write("Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)")
    st.write("https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm")
    st.write("\n \n")
    st.write("\n \n")
    st.write("Kaggle-Data:")
    st.write("https://www.kaggle.com/shayanfazeli/heartbeat")

