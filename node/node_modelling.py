import sys 
import os

# Setup paths
PROJECT_ROOT =  os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from node.src import load_data
from sklearn.model_selection import train_test_split
from node.src import NODECMB

def fit_and_eval(la_trees, la_layers, n_trees, tree_depth, n_layers, X, y, X_test, y_test, lr, st_y, epochs, seed, ovp): 
    tf.keras.utils.set_random_seed(seed = seed)
    
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(lr, 1000)
    node = NODECMB(n_layers = n_layers, n_trees=n_trees, tree_depth=tree_depth, la_trees = la_trees, la_layers= la_layers, ovp = ovp)
    node.compile(loss = "mse", optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, 
                                                                 nesterov = True, momentum = 0.9))    
    if ovp == True:
        #if la_trees > 0: 
        node.fit(X,y, epochs = epochs, batch_size = 1024)
        #else: 
        #    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True)
        #    node.fit(X,y, epochs = epochs, batch_size = 1024, validation_split = 0.2, callbacks = [callback])


        weights_trees_1 = []
        for l in range(n_layers): 
            for t in range(n_trees):
                weights_trees_2 = node.get_layer('aggr_trees_' + str(l)).get_weights()[0].tolist()
                weight = node.get_layer("dense_tree_" + str(l) + "_" + str(t)).get_weights()[0][0][0]
                weights_trees_1.append(weight)
    else:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True)
        node.fit(X,y, epochs = epochs, batch_size = 1024, validation_split = 0.2, callbacks = [callback])
        
        weights_trees_1 = []
        for l in range(n_layers): 
            for t in range(n_trees):
                weights_trees_2 = node.get_layer('aggr_trees_' + str(l)).get_weights()[0].tolist()
                weight = node.get_layer("dense_tree_" + str(l) + "_" + str(t)).get_weights()[0][0][0]
                weights_trees_1.append(weight)

     
    # performance on train/test set
    preds = node.predict(X_test)
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    rmse_rescaled = np.sqrt(np.mean((st_y.inverse_transform(preds) - st_y.inverse_transform(y_test)) ** 2))
    
    record = {
        "seed": seed,
        "ovp" : ovp,
         "la_trees" : la_trees, 
         "la_layers" : la_layers,  
         'n_layers': n_layers, 
         'n_trees': n_trees, 
         'tree_depth': tree_depth, 
         'rmse' : rmse, 
         'rmse_rescaled' : rmse_rescaled,
         'weights_trees_1': weights_trees_1,
         'weigths_trees_2': weights_trees_2
    }
    return record

def benchmark_sparse(la_trees, la_layers, n_trees, tree_depth, n_layers, data_name, lr, epochs, seed, ovp):
    np.random.seed(seed=seed)
    print(la_trees)
    print(la_layers)
    
    data = load_data(data_name)
    X = pd.concat([data[0]['X_train'], data[0]['X_test']])
    y = np.concatenate([data[0]['y_train'], data[0]['y_test']])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)   
    

    st = StandardScaler()
    st_y = StandardScaler()
    X_train = st.fit_transform(X_train)
    y_train = st_y.fit_transform(pd.DataFrame(y_train))
    X_test = st.transform(X_test)
    y_test = st_y.transform(pd.DataFrame(y_test))
        
    result = fit_and_eval(la_trees, la_layers, n_trees, tree_depth, n_layers, X_train, y_train, X_test, y_test, lr, st_y, epochs, seed, ovp)
    
    result_df = pd.DataFrame.from_dict([result]) 
    return result_df
        
        

#  create lambda space
lambda_min = 1e-4
#lambda_min = 0.0014
lambda_max = 0.01
lambda_seq_len = 10
lambda_seq = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), lambda_seq_len))
lambda_seq = np.concatenate([lambda_seq, [0]])

# exemplary call
results = pd.DataFrame()
ovp = True
for l, la_trees in enumerate(lambda_seq):
        result = benchmark_sparse(la_trees, 0, 5, 4, 1, 'wine', 0.4, 200, 55, ovp)
        results = pd.concat([results, result])    
