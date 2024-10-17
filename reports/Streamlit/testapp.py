# Test-Streamlit-File


import streamlit as st
import plotly.graph_objs as go
import numpy as np



import wfdb                            # Package for loading the ecg and annotation
from wfdb import rdrecord, rdann, processing
from sklearn import preprocessing
import numpy as np

import matplotlib.pyplot as plt

data = '..\\..\\data\\.mitdb_original_data\\'
patients = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

dic = {0: ['N', 'L', 'R', 'B'],
       1: ['A', 'a', 'j', 'S', 'e', 'j', 'n'],  # <---- There are 2 small "j"s. Should one of them be "J"??? There is also no small "n"
       2: ['V', 'r', 'E'],
       3: ['F'],
       4: ['Q', '?', 'f', '/'],
      }

def get_ecg_data(record_number = "200"):
    record = wfdb.rdrecord(data+record_number,smooth_frames=True)
    l2 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0]))
    v5 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1]))
    l2 = l2 / np.max(l2)
    v5 = v5 / np.max(v5)
    return l2, v5


def seperate_single_heartbeat(record_number, l2, v5, area):
    area = int(area)
    record_length = 630
    percent_left = 0.4
    percent_right = 1-percent_left
    peak_pos = np.int32(record_length * 0.4)
    max_length_left = np.int32(record_length * percent_left)
    max_length_right = np.int32(record_length * percent_right)

    search_area = 750
    insize=360
    qrs = processing.XQRS(sig = l2[area-search_area:area+search_area],fs = 360)
    qrs.detect()
    peaks = qrs.qrs_inds
    i_closest_peak = np.abs(peaks-search_area).argmin()

    peak_l = peaks[i_closest_peak-1] + area - search_area
    peak = peaks[i_closest_peak] + area - search_area
    peak_r = peaks[i_closest_peak+1] + area - search_area

    # 40% des Signals vor dem Peak
    dleft = np.int32((peak - peak_l) * percent_left)
    if dleft > max_length_left:  dleft = max_length_left 

    # 60% des Signals nach dem Peak
    dright = np.int32((peak_r - peak) * percent_right)
    if dright > max_length_right: dright = max_length_right 

    start = np.int32(peak - dleft)
    end = np.int32(peak + dright)

    ann = wfdb.rdann(data+record_number,extension='atr', sampfrom = start, sampto = end, 
                            return_label_elements=['symbol'])
    temp_classification = ann.symbol[0]

    AAMI = ['N','L','R','B','A','a','j','S','V','r','F','e','j','n','E','f','/','Q','?']
    dic = {'N': ['N', 'L', 'R', 'B'],
        'S': ['A', 'a', 'j', 'S', 'e', 'j', 'n'],  # <---- There are 2 small "j"s. Should one of them be "J"??? There is also no small "n"
        'V': ['V', 'r', 'E'],
        'F': ['F'],
        'Q': ['Q', '?', 'f', '/'],
        }
    for cl, an in dic.items():
        if temp_classification in an:
            classification = cl # Add target variable (the classes)
    class_labels = {"N": 0, 
                    "S": 1,
                    "V": 2,
                    "F": 3,
                    "Q": 4}
    classification_number = class_labels[classification]


    # Stellt sicher, dass sich der Peak immer an der selben Position befindet!
    l2_sig = np.zeros(record_length)
    l2_sig[peak_pos-dleft:peak_pos] = l2[start:peak]
    l2_sig[peak_pos:peak_pos+dright] = l2[peak:end]
    l2_sig = l2_sig[::2]
    xaxis = np.arange(peak-int(record_length*percent_left), peak + int(record_length*percent_right))[::2]

    # Stellt sicher, dass sich der Peak immer an der selben Position befindet!
    v5_sig = np.zeros(record_length)
    v5_sig[peak_pos-dleft:peak_pos] = v5[start:peak]
    v5_sig[peak_pos:peak_pos+dright] = v5[peak:end]
    v5_sig = v5_sig[::2]

    return l2_sig, v5_sig, xaxis, classification_number

pages = ["Introduction", "Datasets", "Data Exploration", 
          "Modelling", "Modelling Results", "Live Test", "Conclusion", "About"]


css='''
<style>
    section.main > div {max-width:100rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)


st.title("Heartbeat Arrhythmia Detection")
st.sidebar.title("Table of contents")
page = st.sidebar.radio("Go to", pages)

if page == "Live Test":
    record_number = st.selectbox("Select a record:", patients)

    l2, v5 = get_ecg_data(record_number)


    x = np.arange(0, len(l2))
    # Erstelle eine Plotly-Grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=l2, name="L2"))
    fig.add_trace(go.Scatter(x=x, y=v5-1, name="V5 - 1"))

    # Zeige die Grafik in Streamlit
    # point = st.plotly_chart(fig, selection_mode="points", on_select="rerun")
    # st.write(point["selection"]["points"][0]["x"])

    fig.update_layout(
        title='Scrollable X-Axis Example',
        xaxis_title='Indices',
        yaxis_title='ECG-Channel / a.u.',
        xaxis=dict(
            rangeslider=dict(visible=True),  # Enable the range slider
            type='linear'  # Ensure the x-axis is linear
        ),
        height=750, width=1200)
    point = st.plotly_chart(fig, selection_mode="box", on_select="rerun")

    area = np.mean(point["selection"]["box"][0]["x"])
    if isinstance(area, (int, float)):

        col1, col2 = st.columns(2, vertical_alignment="center")

        l2_sig, v5_sig, x_heartbeat, classification = seperate_single_heartbeat(record_number, l2, v5, area=area)
        fig = go.Figure()
        fig = fig.add_trace(go.Scatter(x=x_heartbeat, y=l2_sig, name="L2"))
        fig = fig.add_trace(go.Scatter(x=x_heartbeat, y=v5_sig-1, name="V5 - 1"))
        fig.update_layout(
            title='Seperated Heartbeat',
            xaxis_title='Indices',
            yaxis_title='ECG-Channel / a.u.',
            height=400, 
            width=1000)

       
        from tensorflow.keras.models import load_model

        model = load_model("..\\..\\models\\combined_cnn1_bs256_k7_noweights.keras", compile=False)
        prediction = model.predict([np.atleast_2d(l2_sig), np.atleast_2d(v5_sig)])
        text = "  ".join([f"<br>  - Class {i}: {np.round(prediction[0, i],3)} " for i in range(5)])
        # if np.argmax(prediction) == classification:
        #     text = text + "<b><c>**Success**"
        # else:
        #     text = text + "<b>**Failure**"


        # fig.add_annotation(
        #     x=(x_heartbeat[-1] - x_heartbeat[0])*0.7 + x_heartbeat[0],
        #     y=np.max(l2_sig)*0.7,
        #     text=text,
        #     showarrow=False,
        #     align='left',
        #     # ax=0,
        #     # ay=-40
        # )

        col1.plotly_chart(fig)
        col2.write("### Classification")
        col2.markdown(f"Cardiologist: <b>{classification}</b> <br> Prediction-Propability:" + text, unsafe_allow_html=True)

        # col2.markdown(f"<h2 style='font-size: 16px;'> Cardiologist: <b>{classification}</b> <br> Prediction-Propability: <br>" + text + "</h2>", unsafe_allow_html=True)
        if np.argmax(prediction) == classification:
            # st.markdown("<h1 style='font-size: 36px;'>This is a large header</h1>", unsafe_allow_html=True)

            col2.markdown("<h1 style='font-size: 20px; color: green; font-weight: bold'> --- Success ---</h1>", unsafe_allow_html=True)
        else:
            col2.markdown("<h1 style='font-size: 20px; color: red; font-weight: bold'> --- Failure ---</h1>", unsafe_allow_html=True)


        