import tensorflow as tf
from tensorflow.keras import Sequential,utils
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K
from keras import metrics


# Metrics: changed from reset_states to reset_state

class Networks():

    def __init__(self, input=315):
        import tensorflow as tf
        from tensorflow.keras import Sequential,utils
        from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, MaxPool1D, Dropout
        # other imports....

        self.input = input
        self.n_classes = 5
        # self.metrics = ["categorical_accuracy", self.f1_class0, self.f1_class1, self.f1_class2, self.f1_class3, self.f1_class4]
        self.loss = "categorical_crossentropy"
        self.workers = -1
        self.optimizer = "adam"
    
        class F1Score(tf.keras.metrics.Metric):
            def __init__(self, class_index, name='f1_score', **kwargs):
                super(F1Score, self).__init__(name=name, **kwargs)
                self.class_index = class_index
                self.true_positives = self.add_weight(name='tp', initializer='zeros')
                self.false_positives = self.add_weight(name='fp', initializer='zeros')
                self.false_negatives = self.add_weight(name='fn', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_pred = K.round(y_pred)
                
                tp = K.sum(K.round(K.clip(y_true[:, self.class_index] * y_pred[:, self.class_index], 0, 1)))
                fp = K.sum(K.round(K.clip((1 - y_true[:, self.class_index]) * y_pred[:, self.class_index], 0, 1)))
                fn = K.sum(K.round(K.clip(y_true[:, self.class_index] * (1 - y_pred[:, self.class_index]), 0, 1)))
                
                self.true_positives.assign_add(tp)
                self.false_positives.assign_add(fp)
                self.false_negatives.assign_add(fn)

            def result(self):
                precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
                recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
                return 2 * (precision * recall) / (precision + recall + K.epsilon())

            def reset_state(self):
                self.true_positives.assign(0)
                self.false_positives.assign(0)
                self.false_negatives.assign(0)


        class Precision(tf.keras.metrics.Metric):
            def __init__(self, class_index, name='precision', **kwargs):
                super(Precision, self).__init__(name=name, **kwargs)
                self.class_index = class_index
                self.true_positives = self.add_weight(name='tp', initializer='zeros')
                self.false_positives = self.add_weight(name='fp', initializer='zeros')
                self.false_negatives = self.add_weight(name='fn', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_pred = K.round(y_pred)
                
                tp = K.sum(K.round(K.clip(y_true[:, self.class_index] * y_pred[:, self.class_index], 0, 1)))
                fp = K.sum(K.round(K.clip((1 - y_true[:, self.class_index]) * y_pred[:, self.class_index], 0, 1)))
                fn = K.sum(K.round(K.clip(y_true[:, self.class_index] * (1 - y_pred[:, self.class_index]), 0, 1)))
                
                self.true_positives.assign_add(tp)
                self.false_positives.assign_add(fp)
                self.false_negatives.assign_add(fn)

            def result(self):
                precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
                return precision

            def reset_state(self):
                self.true_positives.assign(0)
                self.false_positives.assign(0)
                self.false_negatives.assign(0)

        class Recall(tf.keras.metrics.Metric):
            def __init__(self, class_index, name='recall', **kwargs):
                super(Recall, self).__init__(name=name, **kwargs)
                self.class_index = class_index
                self.true_positives = self.add_weight(name='tp', initializer='zeros')
                self.false_positives = self.add_weight(name='fp', initializer='zeros')
                self.false_negatives = self.add_weight(name='fn', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_pred = K.round(y_pred)
                
                tp = K.sum(K.round(K.clip(y_true[:, self.class_index] * y_pred[:, self.class_index], 0, 1)))
                fp = K.sum(K.round(K.clip((1 - y_true[:, self.class_index]) * y_pred[:, self.class_index], 0, 1)))
                fn = K.sum(K.round(K.clip(y_true[:, self.class_index] * (1 - y_pred[:, self.class_index]), 0, 1)))
                
                self.true_positives.assign_add(tp)
                self.false_positives.assign_add(fp)
                self.false_negatives.assign_add(fn)

            def result(self):
                recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
                return recall

            def reset_state(self):
                self.true_positives.assign(0)
                self.false_positives.assign(0)
                self.false_negatives.assign(0)

        self.metrics = [metrics.CategoricalAccuracy(name="Cat_accuracy"),
                        F1Score(class_index=0, name="F1_score_class0"),
                        F1Score(class_index=1, name="F1_score_class1"),
                        F1Score(class_index=2, name="F1_score_class2"),
                        F1Score(class_index=3, name="F1_score_class3"),
                        F1Score(class_index=4, name="F1_score_class4"),
                        Precision(class_index=0, name="Precision_class0"),
                        Precision(class_index=1, name="Precision_class1"),
                        Precision(class_index=2, name="Precision_class2"),
                        Precision(class_index=3, name="Precision_class3"),
                        Precision(class_index=4, name="Precision_class4"),
                        Recall(class_index=0, name="Recall_class0"),
                        Recall(class_index=1, name="Recall_class1"),
                        Recall(class_index=2, name="Recall_class2"),
                        Recall(class_index=3, name="Recall_class3"),
                        Recall(class_index=4, name="Recall_class4"),
                        ]

    
    def model_dense1(self, input=None):
        """ Just something to implement the class structure"""

        if input is not None:
            self.input = input

        model = Sequential(name="Dense_1")
        model.add(Input(shape=(self.input,), name="Inputlayer"))
        model.add(Dense(units=self.input, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=630, activation="relu", kernel_initializer="normal"))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=630, activation="relu", kernel_initializer="normal"))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=315, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=150, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=self.n_classes, activation="softmax", kernel_initializer="normal"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        model.combined_model = False
        return model
    
    def model_dense2(self, input=None):
        """ Dense2-Struktur """

        if input is not None:
            self.input = input

        model = Sequential(name="Dense_2")
        model.add(Input(shape=(self.input,), name="Inputlayer"))
        model.add(Dense(units=630, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=1260, activation="relu", kernel_initializer="normal"))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=1260, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=1260, activation="relu", kernel_initializer="normal"))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=630, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=630, activation="relu", kernel_initializer="normal"))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=315, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=315, activation="relu", kernel_initializer="normal"))
        model.add(Dense(units=self.n_classes, activation="softmax", kernel_initializer="normal"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        model.combined_model = False
        return model



    
    def model_convolution_smallkernel(self, input=None):
        """ Just something to implement the class structure"""

        if input is not None:
            self.input = input

        model = Sequential(name="Convolution_smallkernel")
        # series_input = Input(shape = (self.input,1))
        model.add(Input(shape = (self.input,1), name="InputLayer"))
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding="valid", activation="relu", name="Conv1_32x3_s1"))
        # model.add(Dropout(0.2, name="Dropout1_p0.2"))
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding="valid", activation="relu", name="Conv2_32x3_s1"))
        model.add(MaxPool1D(pool_size=2, name="MaxPool1_2"))

        model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu", name="Conv3_64x3_s1"))
        # model.add(Dropout(0.2, name="Dropout2_p0.2"))
        model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu", name="Conv4_64x3_s1"))
        model.add(MaxPool1D(pool_size=2, name="MaxPool2_2"))

        model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding="valid", activation="relu", name="Conv5_128x3_s1"))
        # model.add(Dropout(0.2, name="Dropout3_p0.2"))
        model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding="valid", activation="relu", name="Conv6_128x3_s1"))
        model.add(MaxPool1D(pool_size=2, name="MaxPool3_2"))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        model.combined_model = False
        return model


    def model_convolution_bigkernel(self, input=None):
        """ Just something to implement the class structure"""

        if input is not None:
            self.input = input

        model = Sequential(name="Convolution_bigkernel")
        # series_input = Input(shape = (self.input,1))
        model.add(Input(shape = (self.input,1), name="InputLayer"))
        model.add(Conv1D(filters=16, kernel_size=128, strides=2, padding="same", activation="relu", name="Conv1_32x3_s1"))
        model.add(Dropout(0.2, name="Dropout1_p0.2"))
        model.add(Conv1D(filters=16, kernel_size=128, strides=2, padding="same", activation="relu", name="Conv2_32x3_s1"))
        model.add(MaxPool1D(pool_size=2, name="MaxPool1_2"))

        model.add(Conv1D(filters=32, kernel_size=64, strides=2, padding="same", activation="relu", name="Conv3_64x3_s1"))
        model.add(Dropout(0.2, name="Dropout2_p0.2"))
        model.add(Conv1D(filters=32, kernel_size=64, strides=2, padding="same", activation="relu", name="Conv4_64x3_s1"))
        model.add(MaxPool1D(pool_size=2, name="MaxPool2_2"))

        model.add(Conv1D(filters=64, kernel_size=32, strides=2, padding="same", activation="relu", name="Conv5_128x3_s1"))
        model.add(Dropout(0.2, name="Dropout3_p0.2"))
        model.add(Conv1D(filters=64, kernel_size=32, strides=2, padding="same", activation="relu", name="Conv6_128x3_s1"))
        model.add(MaxPool1D(pool_size=2, name="MaxPool3_2"))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        model.combined_model = False
        return model
    

    def model_combined_dense_1(self, input=None):

        if input is not None:
            self.input = input

        def create_model_1(name):
            model = Sequential()
            model.add(Input(shape=(self.input,)))
            model.add(Dense(640, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(630, activation='relu'))
            model.add(Dense(630, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(315, activation='relu'))
            model.add(Dense(315, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            return model

        # Inputlayers
        input_1 = Input(shape=(315, ))
        input_2 = Input(shape=(315, ))

        model_1 = create_model_1(name="ECG_Channel_1")(input_1)
        model_2 = create_model_1(name="ECG_Channel_2")(input_2)

        # Combination
        combined = layers.concatenate([model_1, model_2])
        z = Dense(128, activation='relu')(combined)
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.2)(z)
        z = Dense(32, activation='relu')(z)
        output = Dense(units=self.n_classes, activation='softmax')(z)

        final_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        final_model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        # set name of Model
        final_model._name = "Combined_Dense_1"
        final_model.combined_model = True
        return final_model


    def model_combined_dense_2(self, input=None, kernel_size=3):

        if input is not None:
            self.input = input

        def create_model_1(kernel_size, name):
            model = Sequential(name=name)
            model.add(Input(shape=(self.input,)))
            model.add(Dense(630, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(630, activation='relu'))
            model.add(Dense(630, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1260, activation='relu'))
            model.add(Dense(1260, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(630, activation='relu'))
            model.add(Dense(630, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(315, activation='relu'))
            model.add(Dense(315, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            return model

        # Inputlayers
        input_1 = Input(shape=(315, ))
        input_2 = Input(shape=(315, ))

        model_1 = create_model_1(kernel_size=kernel_size, name="ECG_Channel_1")(input_1)
        model_2 = create_model_1(kernel_size=kernel_size, name="ECG_Channel_2")(input_2)

        # Combination
        combined = layers.concatenate([model_1, model_2])
        z = Dense(128, activation='relu')(combined)
        z = Dense(256, activation='relu')(z)
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.2)(z)
        z = Dense(64, activation='relu')(z)
        output = Dense(units=self.n_classes, activation='softmax')(z)

        final_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        final_model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        # set name of Model
        final_model._name = "Combined_Dense_2"
        final_model.combined_model = True
        return final_model

    def model_combined_cnn_1(self, input=None, kernel_size=3):

        if input is not None:
            self.input = input

        def create_model_1(kernel_size, name):
            model = Sequential(name=name)
            # series_input = Input(shape = (self.input,1))
            model.add(Input(shape = (self.input,1), name="InputLayer"))
            model.add(Conv1D(filters=32, kernel_size=kernel_size, strides=1, padding="valid", activation="relu"))
            # model.add(Dropout(0.2))
            model.add(Conv1D(filters=32, kernel_size=kernel_size, strides=1, padding="valid", activation="relu"))
            model.add(MaxPool1D(pool_size=2))

            model.add(Conv1D(filters=64, kernel_size=kernel_size, strides=1, padding="valid", activation="relu"))
            # model.add(Dropout(0.2"))
            model.add(Conv1D(filters=64, kernel_size=kernel_size, strides=1, padding="valid", activation="relu"))
            model.add(MaxPool1D(pool_size=2))

            model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=1, padding="valid", activation="relu"))
            # model.add(Dropout(0.2))
            model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=1, padding="valid", activation="relu"))
            model.add(MaxPool1D(pool_size=2))

            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            return model

        # Inputlayers
        input_1 = Input(shape=(315, ))
        input_2 = Input(shape=(315, ))

        model_1 = create_model_1(kernel_size=kernel_size, name="ECG_Channel_1")(input_1)
        model_2 = create_model_1(kernel_size=kernel_size, name="ECG_Channel_2")(input_2)

        # Combination
        combined = layers.concatenate([model_1, model_2])
        z = Dense(256, activation='relu')(combined)
        z = Dense(256, activation='relu')(z)
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.3)(z)
        z = Dense(64, activation='relu')(z)
        output = Dense(units=self.n_classes, activation='softmax')(z)

        final_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001/2)
        final_model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        # set name of Model
        final_model._name = "Combined_cnn_1"
        final_model.combined_model = True
        return final_model
