# %% Libary Imports: 

# run with: streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Data Import
# ....

# %% Streamlit:

pages = ["Introduction", "Datasets", "Data Exploration and Visualization", 
          "Modelling", "Modelling Results", "Live Test", "Conclusion", "About"]

st.title("Heartbeat Arrhythmia Detection")
st.sidebar.title("Table of contents")
page = st.sidebar.radio("Go to", pages)

st.sidebar.write("# Participants: \nFelix Brand")

if page == "Introduction":  # Match with "pages"-entries
    st.write(f"## {page}")
    st.image("figures\Streamlit\Heartbeat_scheme.png")
    st.write("We will write something medical perspective. ")
    st.write("PQRST-Scheme")
    st.write("Classes 0...4 with meaning ")


if page == "Datasets":  # Match with "pages"-entries
    st.write(f"## {page}")
    st.write("Comparision between two kinds of datasets")
    dataset_radio = st.radio("Dataset:", ("Kaggle", "Original"))
    if dataset_radio == "Kaggle":
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(np.arange(1,10))
        st.pyplot(fig)

    if dataset_radio == "Original":
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(-np.arange(1,10))
        st.pyplot(fig)
    
    measurement_setup = {
        "Measurement type": "2-channel ECG á 30 min",
        "Device": "Del-Mar Avionics model 445",
        "Sampling frequency": "360 Hz",
        "Frequency filter": "0.1 ... 100 Hz",
        "Measurement resolution": "11 bit",
        "Input voltage": "+- 10 mV",
        "Patients": "47 patients, 48 records",
        "Additional information": "Age, Medication, Sex", 
        "Classification": "2 indiviual cardiologists, annotations at the R-Peak"
                        }
    st.write("### Measurement-Setup")
    st.dataframe(pd.DataFrame({"Setting": measurement_setup.keys(), 
              "Value": measurement_setup.values()}), hide_index=True)

    colname1, colname2, colname3 = ["Setting", "Kaggle", "Original"]
    dataset_comparision = pd.DataFrame([
        {colname1: "Sampling-Rate", colname2: "125 Hz - Interpolation", colname3: "180 - undersampling"},
        {colname1: "Normalisation", colname2: "Min-Max-Scaling per Heartbeat", colname3: "Max-Scaling per patient"}, 
        {colname1: "Data", colname2: "One channel / unknown", colname3: "Both channels"},
        {colname1: "Seperation", colname2: "R to T", colname3: "P to T"},
        {colname1: "...", colname2: "...", colname3: "..."},
                                        ])
    st.dataframe(dataset_comparision, hide_index=True)

    st.write("## Conclusion:")
    st.write("use of the original data")


if page == "Data Exploration and Visualization":
    st.write(f"## {page}")
    st.write("- Amount of data / heartbeats")
    st.write("- amount of 0-Data ")
    st.write("- unbalanced Dataset - short ")
    st.write("- Figure with 1 or 2 random signals of each class")
    

if page == "Modelling":
    st.write(f"## {page}")
    st.write("- show different kinds of Neuronal Networks")
    st.write("- Mention different kinds of preprocessing (FFT, substract-mean, digital filter)")
    st.write("- Mention different kinds of weights")
    st.write("")

    choice = ["Dense 1", "Dense 2", "CNN 1 - small kernel", "CNN 2 - big kernel", "Fusion Dense", "Fusion CNN"]
    selection = st.selectbox('Model', choice)
    if selection == choice[0]:
        st.image("figures\Streamlit\dummy_network.jpg")
    if selection == choice[1]:
        st.image("figures\Streamlit\dummy_network.jpg")
    if selection == choice[2]:
        st.image("figures\Streamlit\dummy_network.jpg")


if page == "Modelling Results":
    st.write(f"## {page}")
    st.write("Mention parameter grid: \n "
              "- Model \n"
              "- Batch size: 256, 512, 1024, 2048 \n"
              "- Class weights normal, balanced, sigmoid \n"
              "- Kernel size: 3, 5, 7\n" 
              "- Data: Channel1, Channel2, [Channel1, Channel2], [Channel1 | Channel 2]")
    st.write("### Boxplots for different kinds of hyperparameter")
    col1, col2, col3, col4 = st.columns(4)
    col1.radio("Parameter:", ("Model", "Batch size", "class weights", "Data"))
    col2.radio("Dataset:", ("Train", "Validation"))
    col3.radio("Metric:", ("Accuracy", "Recall", "F1"))
    col4.radio("Class:", ("Overall", "Class 0", "Class 1", "Class 2", "Class 3", "Class 4"))


    fig, ax = plt.subplots()
    data = np.random.rand(1000, 5)
    ax.boxplot(data)
    ax.set_title("Dummy graph")
    st.pyplot(fig)

    st.write("## Conclusion:")
    st.write("Decision for: ...")

    st.write("Balance Report for that combination and the crosstab")
 
if page == "Live Test":
    st.write("How should the live-test be performed?")
    st.write("30min patient-measurement with slider and perform a classification on a certain area?")


if page == "Conclusion":
    st.write(f"## {page}")
    st.write("Here comes some text for the conclusion")


if page == "About":
    st.write(f"## {page}")
    st.write("Participants:\n"
             "- Felix Brand\n"
             "- ....")
    st.write("Data: \n Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)")



