import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Dense, concatenate, Embedding, Flatten, Dot

def second_order_interaction(num_outputs, cat_outputs):
    
    second_order_outputs = []
    
    for cat in cat_outputs:
        second_order_outputs.append(Dot(axes=-1)([num_outputs, cat]))
    
    for i in range(len(cat_outputs)):
        for j in range(len(cat_outputs)):
            if i != j:
                second_order_outputs.append(Dot(axes=-1)([cat_outputs[i], cat_outputs[j]]))
                
    return second_order_outputs
    

def DLRMModel(num_inputs, 
              cat_inputs,
              feature_dic,
              num_units=[512,256, 16],
              embed_unit = 16,
              layer_units=[512, 256],
              activation='relu'):
    
    numerical_input = Input(shape=(num_inputs.shape[1],), name='Numerical Input')
    categorical_inputs = []
    embedded = []
    
    # One-hot encode the cat input data 
    for i, col in enumerate(cat_inputs):
        data_shape = cat_inputs[[col]].shape[1]
        cat_input = Input(shape=(data_shape,), name=f'Categorical Input_{i}')
        categorical_inputs.append(cat_input)
        #oh_ = tf.one_hot(cat_input, feature_dic[col])
        embed = Embedding(input_dim = feature_dic[col], output_dim=embed_unit)(cat_input)
        embed = Flatten()(embed)
        embedded.append(embed)

    # Dense networkd for numerical data 
    x = Dense(num_units[0], activation=activation)(numerical_input)
    for unit in num_units[1:]:
        x = Dense(unit, activation=activation)(x)
    
    # make inner product combination of embedding and num_value
    second_order_outputs = second_order_interaction(x, embedded)
    
    # concat
    concat = concatenate([second_order for second_order in second_order_outputs], axis=-1)
    concat = concatenate([x, concat], axis=-1)
    
    # dense for concat layer 
    
    x = Dense(layer_units[0], activation=activation)(concat)
    for unit in layer_units[1:]:
        x = Dense(unit, activation=activation)(x)
    
    output = Dense(1, activation='sigmoid', name='Output')(x)
    
    model = Model(inputs=[numerical_input, categorical_inputs], outputs=output )
    
    return model
    
        
        

class NumericalDense(Model):

    def __init__(self, units=[512, 256, 16], activation='relu', **kwargs):
        '''Initializes the class and sets up the internal variables'''
        super().__init__(**kwargs)
        
        self.layers = []
        
        for unit in units:
            layer = Dense(unit, activation=activation)
            self.layers.append(layer)
            
    def call(self, inputs):
        
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        
        return x
    
    
class CategoricalDense(Model):

    def __init__(self, feature_dic, embed_unit=16, activation='relu', **kwargs):
        '''Initializes the class and sets up the internal variables'''
        super().__init__(**kwargs)
        
        self.embed_layers = []
        self.feature_dic = feature_dic
        
        for _, item in feature_dic.items():
            layer = Embedding(item, embed_unit)
            self.embed_layers.append(layer)
            
    def call(self, inputs):
        
        input_oh = []
        output = []
        for idx, _input in enumerate(inputs):
            oh_ = tf.one_hot(_input, self.feature_dic[idx])
            input_oh.append(oh_)
        
        for layer, oh in zip(self.embed_layers, input_oh):
            output.append(layer(oh))

        return output

class DLRM(Model):
    def __init__(self, units=[512, 256], activation='relu', **kwargs):
        '''Initializes the class and sets up the internal variables'''
        super(DLRM, self).__init__(**kwargs)
        
        self.layers = []
        
        for unit in units:
            layer = Dense(unit, activation=activation)
            self.layers.append(layer)
        
        output_layer = Dense(1, activation='softmax')
        self.layers.append(output_layer)
        
    def call(self, inputs):
        pass