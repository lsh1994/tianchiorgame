"""
@file: mapping.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/08
"""
import keras
from keras import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
import BaseModel.pipeline as pipe


def get_model():
    model=keras.models.load_model("output/my_model.h5")
    mapping_model=Model(input=model.input,output=model.layers[-2].output)
    y=Dense(30,activation='sigmoid',name="label_feather")(mapping_model.output)

    my_model=Model(input=mapping_model.input,output=y)
    trainable = False
    for lay in my_model.layers:
        if (lay.name == "label_feather"):
            trainable = True
        lay.trainable = trainable

    my_model.summary()
    my_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(),
                     metrics=['accuracy'])
    return my_model

if __name__ == '__main__':
    labelkeys = ['ZJL1', 'ZJL10', 'ZJL100', 'ZJL101', 'ZJL102', 'ZJL103', 'ZJL104', 'ZJL105', 'ZJL106', 'ZJL107',
                 'ZJL108', 'ZJL109', 'ZJL11', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117',
                 'ZJL118', 'ZJL119', 'ZJL12', 'ZJL120', 'ZJL121', 'ZJL122', 'ZJL123', 'ZJL124', 'ZJL125', 'ZJL126',
                 'ZJL127', 'ZJL128', 'ZJL129', 'ZJL13', 'ZJL130', 'ZJL131', 'ZJL132', 'ZJL133', 'ZJL135', 'ZJL137',
                 'ZJL138', 'ZJL139', 'ZJL14', 'ZJL140', 'ZJL141', 'ZJL142', 'ZJL143', 'ZJL144', 'ZJL145', 'ZJL146',
                 'ZJL147', 'ZJL149', 'ZJL15', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL154', 'ZJL156', 'ZJL157',
                 'ZJL158', 'ZJL159', 'ZJL16', 'ZJL160', 'ZJL161', 'ZJL162', 'ZJL163', 'ZJL164', 'ZJL165', 'ZJL166',
                 'ZJL167', 'ZJL168', 'ZJL169', 'ZJL170', 'ZJL171', 'ZJL172', 'ZJL173', 'ZJL174', 'ZJL175', 'ZJL176',
                 'ZJL177', 'ZJL178', 'ZJL179', 'ZJL18', 'ZJL180', 'ZJL181', 'ZJL182', 'ZJL183', 'ZJL184', 'ZJL185',
                 'ZJL186', 'ZJL187', 'ZJL188', 'ZJL189', 'ZJL19', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194',
                 'ZJL195', 'ZJL2', 'ZJL21', 'ZJL22', 'ZJL23', 'ZJL24', 'ZJL25', 'ZJL26', 'ZJL28', 'ZJL29', 'ZJL3',
                 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL38', 'ZJL39', 'ZJL4', 'ZJL40',
                 'ZJL41', 'ZJL42', 'ZJL43', 'ZJL44', 'ZJL45', 'ZJL46', 'ZJL47', 'ZJL48', 'ZJL49', 'ZJL5', 'ZJL50',
                 'ZJL51', 'ZJL52', 'ZJL53', 'ZJL54', 'ZJL55', 'ZJL56', 'ZJL57', 'ZJL58', 'ZJL59', 'ZJL6', 'ZJL60',
                 'ZJL61', 'ZJL62', 'ZJL63', 'ZJL64', 'ZJL65', 'ZJL66', 'ZJL67', 'ZJL68', 'ZJL69', 'ZJL7', 'ZJL70',
                 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL8', 'ZJL80', 'ZJL81',
                 'ZJL82', 'ZJL83', 'ZJL84', 'ZJL85', 'ZJL86', 'ZJL87', 'ZJL88', 'ZJL89', 'ZJL9', 'ZJL90', 'ZJL91',
                 'ZJL92', 'ZJL93', 'ZJL94', 'ZJL95', 'ZJL96', 'ZJL97', 'ZJL98', 'ZJL99']

    cpi = pipe.DataPiple(target="data/train_img.csv", labelkeys=labelkeys)
    tr_flow = cpi.create_inputs(size=64)


    my_model = get_model()

    checkpoint = ModelCheckpoint(filepath="output/mapping.h5", monitor='acc', mode='auto', save_best_only='True')
    tensorboard = TensorBoard(log_dir='output/log_mapping')
    my_model.fit_generator(tr_flow, steps_per_epoch=32, epochs=1000, verbose=2,callbacks=[tensorboard, checkpoint])