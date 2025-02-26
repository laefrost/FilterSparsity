import tensorflow as tf
#import tensorflow_addons as tfa
from src.activations import sparsemax 
from tensorflow_probability import distributions, stats
import numpy as np
import random
import keras 

def log_to_file(tensor, filename, tensor_name):
    with open(filename, 'a') as f:
        # Convert tensor to a numpy array and write to file
        f.write(tensor_name + ":" + str(tensor.numpy()) + '\n')

@tf.function
def sparsemoid(inputs):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)


def get_binary_lookup_table(depth):
    # output: binary tensor [depth, 2**depth, 2]
    indices = tf.keras.backend.arange(0, 2**depth, 1)
    offsets = 2 ** tf.keras.backend.arange(0, depth, 1)
    bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
    bin_codes = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
    bin_codes = tf.cast(bin_codes, 'float32')
    # binary_lut = tf.Variable(initial_value=bin_codes, trainable=False, validate_shape=False)
    return bin_codes


def get_feature_selection_logits(n_trees, depth, dim):
    initializer = tf.keras.initializers.random_uniform()
    init_shape = (dim, n_trees, depth)
    init_value = initializer(shape=init_shape, dtype='float32')
    return tf.Variable(init_value, trainable=True)


def get_output_response(n_trees, depth, units):
    initializer = tf.keras.initializers.random_uniform()
    init_shape = (n_trees, units, 2**depth)
    init_value = initializer(init_shape, dtype='float32')
    return tf.Variable(initial_value=init_value, trainable=True)


def get_feature_thresholds(n_trees, depth):
    initializer = tf.ones_initializer()
    init_shape = (n_trees, depth)
    init_value = initializer(shape=init_shape, dtype='float32')
    return tf.Variable(init_value, trainable=True)


def get_log_temperatures(n_trees, depth):
    initializer = tf.ones_initializer()
    init_shape = (n_trees, depth)
    init_value = initializer(shape=init_shape, dtype='float32')
    return tf.Variable(initial_value=init_value, trainable=True)


def init_feature_thresholds(features, beta, n_trees, depth):
    sampler = distributions.Beta(beta, beta)
    percentiles_q = sampler.sample([n_trees * depth])

    flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, features)
    percentile = stats.percentile(flattened_feature_values, 100*percentiles_q)
    feature_thresholds = tf.reshape(percentile, (n_trees, depth))
    return feature_thresholds


def init_log_temperatures(features, feature_thresholds):
    input_threshold_diff = tf.math.abs(features - feature_thresholds)
    log_temperatures = stats.percentile(input_threshold_diff, 50, axis=0)
    return log_temperatures

#@keras.saving.register_keras_serializable(package="ObliviousDecisionTree")
class ObliviousDecisionTreeCMB(tf.keras.layers.Layer):
    def __init__(self,
                 n_trees=3,
                 depth=4,
                 units=1,
                 threshold_init_beta=1., initialized = False, seed= 123,  **kwargs):
        super(ObliviousDecisionTreeCMB, self).__init__(**kwargs)
        self.initialized = initialized
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
        self.seed = seed

    # invoked by the first __call__() to the layer, and supplies the shape(s) of the input(s), which may not have been known at initialization time;
    def build(self, input_shape):
        dim = input_shape[-1]
        n_trees, depth, units, seed = self.n_trees, self.depth, self.units, self.seed
        
        # trainable parameters 
        ### initialization via random_uniform(), shape: (dim, n_trees, depth)
        self.feature_selection_logits = self.add_weight(shape= (dim, n_trees, depth), 
                                                        initializer = tf.keras.initializers.RandomNormal(seed = seed), 
                                                        trainable = True)#,
                                                        #regularizer=tf.keras.regularizers.L2())
                                                                     
        ### initialization via tf.ones_initializer(), shape: (n_trees, depth)                                                             
        self.feature_thresholds = self.add_weight(shape= (n_trees, depth), 
                                                        initializer = tf.ones_initializer(), 
                                                        trainable = True) #,
                                                        #regularizer=tf.keras.regularizers.L2())

        ### initialization via tf.ones_initializer(), shape: (n_trees, depth)                                                             
        self.log_temperatures = self.add_weight(shape= (n_trees, depth), 
                                                        initializer = tf.ones_initializer(), 
                                                        trainable = True) #,
                                                        #regularizer=tf.keras.regularizers.L2())


        ### initialization via random_uniform(), (n_trees, units, 2^depth)
        self.response = self.add_weight(shape = (n_trees, units, 2**depth),
                                        initializer = tf.keras.initializers.RandomNormal(seed = seed), 
                                                        trainable = True) #,
                                                        #regularizer=tf.keras.regularizers.L2())


        # non-trainable parameter, binary tensor [depth, 2**depth, 2]
        with tf.init_scope():
            self.binary_lut = tf.Variable(initial_value=get_binary_lookup_table(depth), trainable=False, validate_shape=False)

    
    # assigns starting values to self.feature_thresholds and self.log_temperatures
    def _data_aware_initialization(self, inputs):
        beta, n_trees, depth = self.threshold_init_beta, self.n_trees, self.depth

        feature_values = self._get_feature_values(inputs)
        feature_thresholds = init_feature_thresholds(feature_values, beta, n_trees, depth)
        log_temperatures = init_log_temperatures(feature_values, feature_thresholds)
        
        self.feature_thresholds.assign(feature_thresholds)
        self.log_temperatures.assign(log_temperatures)

    def _get_feature_values(self, inputs, training=None):
        # Probability per feature, per tree, per level of tree (i,n,d)
        feature_selectors = sparsemax(self.feature_selection_logits)
        
        # shape: (batch_size, n_trees, tree_depth)
        # b = batch, i = feature, n = n_tress, d = depth
        # per batch data point, tree und tree level weighted sum of input features with the corresponding weights from feature_selectors
        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        return feature_values

    def _get_feature_gates(self, feature_values):
        threshold_logits = (feature_values - self.feature_thresholds)
        threshold_logits = threshold_logits * tf.math.exp(-self.log_temperatures)
        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)

        feature_gates = sparsemoid(threshold_logits)
        return feature_gates

    def _get_aggregated_response(self, feature_gates):
        # b: batch, n: number of trees, d: depth of trees, s: 2 (binary channels)
        # c: 2**depth, u: units (response units)
        
        aggregated_gates = tf.einsum('bnds,dcs->bndc', feature_gates, self.binary_lut)
        aggregated_gates = tf.math.reduce_prod(aggregated_gates, axis=-2)
       
        # response per batch data point, tree and unit
        # weighted linear combination of response tensor entries with weights from the entries of choice tensor C
        # shape (b, n, u)
        aggregated_response = tf.einsum('bnc,nuc->bnu', aggregated_gates, self.response)
        return aggregated_response

 
    def call(self, inputs, training=None):
        if not self.initialized:
            self._data_aware_initialization(inputs)
            self.initialized = True
        feature_values = self._get_feature_values(inputs)
        feature_gates = self._get_feature_gates(feature_values)
        aggregated_response = self._get_aggregated_response(feature_gates)
        
        # shape: (b, u)
        #response_averaged_over_trees = tf.reduce_mean(aggregated_response, axis=1)
        return aggregated_response
      
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initialized' : self.initialized, 
            'n_trees': self.n_trees,
            'depth': self.depth, 
            'units': self.units, 
            'threshold_init_beta': self.threshold_init_beta
        })
        return config
      
#@keras.saving.register_keras_serializable(package="NODE")
class NODECMB(tf.keras.Model):
    def __init__(self,
                 units = 1,
                 n_layers = 1,
                 n_trees = 1,
                 tree_depth = 1,
                 threshold_init_beta = 1,
                 la_trees = 0,
                 #la_layers = 0, 
                 #bn = None, ensemble = None, 
                 ovp = True, seed = 123, **kwargs):

        super(NODECMB, self).__init__(**kwargs)
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.threshold_init_beta = threshold_init_beta
        self.la_trees = la_trees
        #self.la_layers = la_layers
        self.bn = tf.keras.layers.BatchNormalization()
        self.concat_tree = tf.keras.layers.Concatenate()
        self.ovp = ovp
        
        # 2nd overparametrization across trees
        self.aggr_trees_per_layer = [tf.keras.layers.Dense(1, use_bias = False, 
                                             kernel_regularizer = tf.keras.regularizers.L2(self.la_trees), 
                                             name = "aggr_trees_" + str(_), 
                                             kernel_initializer=tf.keras.initializers.glorot_normal(seed = seed))
                           for _ in range(n_layers)]
        
        self.trees_per_layer = []
        # 1st overparametrization per tree 
        for l in range(n_layers): 
            dense_regu_trees = [tf.keras.layers.Dense(1, use_bias = False, name = "dense_tree_" + str(l) + "_" +str(_),
                                                 kernel_regularizer = tf.keras.regularizers.L2(self.la_trees),
                                                 kernel_initializer=tf.keras.initializers.glorot_normal(seed = seed))
                          for _ in range(n_trees)]
            self.trees_per_layer.append(dense_regu_trees)
        
        #self.dense_regu_layers = [tf.keras.layers.Dense(1, use_bias = False, name = "dense_layer_" + str(_),
        #                                                kernel_regularizer = tf.keras.regularizers.L2(self.la_layers))
        #                          for _ in range(n_layers)]
        
        self.ensemble = [ObliviousDecisionTreeCMB(n_trees=n_trees,
                             depth=tree_depth,
                             units=units,
                             threshold_init_beta=threshold_init_beta, 
                             seed = seed)
                         for _ in range(n_layers)]


# defines forward pass 
# 1) batch normalization of complete input
# 2) call ODT Layers using the input and the output from the previous layer
    def call(self, inputs, training=None):
        x = inputs
        layers_output = []
        for l, layer in enumerate(self.ensemble):
            h = layer(x)
            if self.ovp == True and self.la_trees > 0:
                h_t = tf.transpose(h, [0, 2, 1])
                trees_output = []
                for t, dense in enumerate(self.trees_per_layer[l]):
                    tree = h_t[: , : , t]
                    tree = dense(tree) # first ovp per tree
                    trees_output.append(tree)
                trees_output = self.concat_tree(trees_output) # concat all once ovp'ed treed
                h = self.aggr_trees_per_layer[l](trees_output) # outer ovp across all trees in the respective layer     
            else: 
                h = tf.reduce_mean(h, axis = 1)
            
            # if self.la_layers > 0:             
            #     h = self.dense_regu_layers[l](h)
            
            layers_output.append(h)
            x = tf.concat([x, h], axis=1)
        aggr_layer_outputs = tf.reduce_mean(layers_output, axis=0)
        return aggr_layer_outputs
    
      
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'n_layers': self.n_layers, 
            'tree_depth': self.tree_depth, 
            'threshold_init_beta': self.threshold_init_beta,
            'n_trees' : self.n_trees, 
            'bn' : self.bn, 
            'la_trees' : self.la_trees, 
            'ensemble' : keras.saving.serialize_keras_object(self.ensemble)
            
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("ensemble")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        u = config.pop('units')
        nl = config.pop('n_layers')
        td = config.pop('tree_depth')
        tib  = config.pop('threshold_init_beta')
        nt = config.pop('n_trees')
        bn = config.pop('bn')
        la_trees =  config.pop('la_trees')
        return cls(u, nl, nt, td, tib, bn, la_trees, sublayer, **config)
      

#def layer_node(units, n_layers, n_trees, tree_depth, threshold_init_beta):
#    return(NODE(units, n_layers, n_trees, tree_depth, threshold_init_beta))