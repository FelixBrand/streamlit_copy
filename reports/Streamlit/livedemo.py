# Test-Streamlit-File

import plotly.graph_objs as go
import numpy as np
import wfdb
from wfdb import rdrecord, rdann, processing
from tensorflow.keras.models import load_model
from sklearn import preprocessing

FONT = dict(family="Arial", size=20, color="black")

class livedemo():
    """ Class for livedemo in streamlit"""
    def __init__(self):

        self.data = '..\\..\\data\\.mitdb_original_data\\'
        self.patients = ['100','101','102','103','104','105','106','107',
                '108','109','111','112','113','114','115','116',
                '117','118','119','121','122','123','124','200',
                '201','202','203','205','207','208','209','210',
                '212','213','214','215','217','219','220','221',
                '222','223','228','230','231','232','233','234']

        self.dic = {0: ['N', 'L', 'R', 'B'],
            1: ['A', 'a', 'j', 'S', 'e', 'j', 'n'],
            2: ['V', 'r', 'E'],
            3: ['F'],
            4: ['Q', '?', 'f', '/'],
            }
        self.model = load_model("..\\..\\models\\combined_cnn1_bs256_k7_noweights.keras",
                                compile=False)


    def get_ecg_data(self, record_number = "200"):
        """ Imports a single 30 min ecg and returns both channels"""
        record = wfdb.rdrecord(self.data+record_number,smooth_frames=True)
        l2 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0]))
        v5 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1]))
        l2 = l2 / np.max(l2)
        v5 = v5 / np.max(v5)
        return l2, v5


    def seperate_single_heartbeat(self, record_number, area):
        """ Gets a single heartbeat around the 'area' and returns
        both channels for that heartbeat and the class"""
        area = int(area)
        record_length = 630
        percent_left = 0.4
        percent_right = 1-percent_left
        peak_pos = np.int32(record_length * 0.4)
        max_length_left = np.int32(record_length * percent_left)
        max_length_right = np.int32(record_length * percent_right)

        search_area = 750
        qrs = processing.XQRS(sig = self.l2[area-search_area:area+search_area],fs = 360)
        qrs.detect()
        peaks = qrs.qrs_inds
        i_closest_peak = np.abs(peaks-search_area).argmin()

        peak_l = peaks[i_closest_peak-1] + area - search_area
        peak = peaks[i_closest_peak] + area - search_area
        peak_r = peaks[i_closest_peak+1] + area - search_area

        # 40% des Signals vor dem Peak
        dleft = np.int32((peak - peak_l) * percent_left)
        dleft = min(dleft, max_length_left)
        # if dleft > max_length_left:  dleft = max_length_left

        # 60% des Signals nach dem Peak
        dright = np.int32((peak_r - peak) * percent_right)
        dright = min(dright, max_length_right)
        # if dright > max_length_right: dright = max_length_right

        start = np.int32(peak - dleft)
        end = np.int32(peak + dright)

        ann = wfdb.rdann(self.data+record_number,extension='atr', sampfrom = start, sampto = end,
                                return_label_elements=['symbol'])
        temp_classification = ann.symbol[0]

        # AAMI = ['N','L','R','B','A','a','j','S','V','r','F','e','j','n','E','f','/','Q','?']
        dic = {'N': ['N', 'L', 'R', 'B'],
            'S': ['A', 'a', 'j', 'S', 'e', 'j', 'n'],
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
        l2_sig[peak_pos-dleft:peak_pos] = self.l2[start:peak]
        l2_sig[peak_pos:peak_pos+dright] = self.l2[peak:end]
        l2_sig = l2_sig[::2]
        xaxis = np.arange(peak-int(record_length*percent_left),
                          peak + int(record_length*percent_right))[::2]

        # Stellt sicher, dass sich der Peak immer an der selben Position befindet!
        v5_sig = np.zeros(record_length)
        v5_sig[peak_pos-dleft:peak_pos] = self.v5[start:peak]
        v5_sig[peak_pos:peak_pos+dright] = self.v5[peak:end]
        v5_sig = v5_sig[::2]

        return l2_sig, v5_sig, xaxis, classification_number

    def record_plot(self, record_number):
        """ imports one record and plot it"""
        l2, v5 = self.get_ecg_data(record_number)
        self.l2 = l2
        self.v5 = v5

        x = np.arange(0, len(l2))
        # Erstelle eine Plotly-Grafik
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=l2, name="L2"))
        fig.add_trace(go.Scatter(x=x, y=v5-1, name="V5 - 1"))

        # Zeige die Grafik in Streamlit
        # point = st.plotly_chart(fig, selection_mode="points", on_select="rerun")
        # st.write(point["selection"]["points"][0]["x"])

        fig.update_layout(
            title = dict(text=f'30 min ECG, record {record_number}', font=dict(size=20)),
            xaxis_title=dict(text='Indices', font=FONT),
            yaxis_title=dict(text='ECG-Channel / a.u.', font=FONT),
            xaxis = dict(
                rangeslider = dict(visible=True),  # Enable the range slider
                type='linear'  # Ensure the x-axis is linear
            ),
            height = 750, width = 1200)

        return fig


    def single_heartbeat_plot(self, record_number, area):
        """ Returns the plotly plot of a single heartbeat and both channels"""
        if isinstance(area, (int, float)):
            l2_sig, v5_sig, x_beat, real_classification = self.seperate_single_heartbeat(record_number,
                                                                                         area=area)
            fig = go.Figure()
            fig = fig.add_trace(go.Scatter(x=x_beat, y=l2_sig, name="L2"))
            fig = fig.add_trace(go.Scatter(x=x_beat, y=v5_sig-1, name="V5 - 1"))

            fig.update_layout(
                title=dict(text='Seperated Heartbeat', font=dict(size=20)),
                xaxis_title=dict(text='Indices', font=FONT),
                yaxis_title=dict(text='ECG-Channel / a.u.', font=FONT),
                height=400,
                width=1000)

        return fig, l2_sig, v5_sig, real_classification


    def prediction(self, l2_sig, v5_sig, real_classification):
        """ Predition of the single heartbeat
        returns the text for the classification"""

        prediction = self.model.predict([np.atleast_2d(l2_sig), np.atleast_2d(v5_sig)])
        text = "  ".join([f"<br>  - Class {i}: {np.round(prediction[0, i],3)} " for i in range(5)])

        text1 = "### Classification"
        text2 = f"Cardiologist: <b>{real_classification}</b> <br> Prediction-Propability:" + text
        if np.argmax(prediction) == real_classification:
            text3 = "<h1 style='font-size: 20px; color: green; font-weight: bold'> --- Success ---</h1>"
        else:
            text3 = "<h1 style='font-size: 20px; color: red; font-weight: bold'> --- Failure ---</h1>"
        return text1, text2, text3

