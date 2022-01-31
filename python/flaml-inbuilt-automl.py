from flaml import AutoML
import pandas as pd
import numpy as np
import lightgbm as lgb

#settings
timebudget = 2 * 60 * 60
metric = 'accuracy'
DOWNCAST = True
train_file_path = '../data/train.csv'
validation_file_path = '../data/test.csv'
LABEL = "survived"

#****************************
#Read training data
train_set = pd.read_csv(train_file_path, low_memory=True, warn_bad_lines=True) 
print(train_set)

#MINMIZE: Let's test is downcast makes a different - it should not
if DOWNCAST:
    fcols = train_set.select_dtypes('float').columns
    icols = train_set.select_dtypes('integer').columns
    ocols = train_set.select_dtypes('object').columns

    print("Downcast float")
    train_set[fcols] = train_set[fcols].apply(pd.to_numeric, downcast='float')
    print("Downcast integer")
    train_set[icols] = train_set[icols].apply(pd.to_numeric, downcast='integer')
    print("Downcast object")
    train_set[ocols] = train_set[ocols].apply(pd.Series.astype, dtype='category')

y_train = train_set.pop(LABEL)
print(y_train)


#*****************************
#Read validation data
val_set = pd.read_csv(validation_file_path, low_memory=True, warn_bad_lines=True)
print(val_set)


#minimize
if DOWNCAST:
    fcols = val_set.select_dtypes('float').columns
    icols = val_set.select_dtypes('integer').columns
    ocols = val_set.select_dtypes('object').columns

    val_set[fcols] = val_set[fcols].apply(pd.to_numeric, downcast='float')
    val_set[icols] = val_set[icols].apply(pd.to_numeric, downcast='integer')
    val_set[ocols] = val_set[ocols].apply(pd.Series.astype, dtype='category')

            
y_val = val_set.pop(LABEL)
print(y_val)

train_set.reset_index(inplace=True, drop=True)
val_set.reset_index(inplace=True, drop=True)

               
                
automl = AutoML()
settings = {
    "time_budget": timebudget, 
    "metric": metric,  
    "estimator_list": ['lgbm'],  
    "task": 'binary', 
    "log_file_name": train_file_path + '.log',
    "seed": 123, 
}

print(train_set.size)
print(val_set.size)

automl.fit(X_train=train_set, y_train=y_train, X_val=val_set, y_val=y_val, early_stop=True, **settings)

print(automl.model)
print(automl.model.estimator)
print(automl.best_iteration)
print(automl.best_loss)

f = open( "flaml-best-model.txt", "a")
f.write( str(lgb.__version__) )
f.write("\nLABEL ")
f.write( str(LABEL) )
f.write("\nMETRIC ")
f.write( str(metric) )
f.write("\nBEST LOSS ")
f.write( str(automl.best_loss) )
f.write("\nTIMEBUDGET ")
f.write( str(timebudget) )
f.write("\nTRAININGFILE ")
f.write( str(train_file_path) )
f.write("\n")
f.write( str(automl.model.estimator))
f.write("\n\n")
f.close()
