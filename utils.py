import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.svm import SVR

def xgb_model(epochs, objective, booster, max_depth):
    model = XGBRegressor(n_estimators = epochs,
                         max_depth = max_depth,
                         objective = objective,
                         tree_method = 'gpu_hist',
                         booster = booster,
                         verbosity = 1)
    return model

def xgb_simple_test(df, label):
    
    X_train, X_test, y_train, y_test = train_test_split(df, label,
                                                        test_size=0.33, random_state=42)

    max_score = 0

    
    # model training 
    model = xgb_model(200, 'reg:squarederror', 'gbtree', 8) # epochs, objective, booster, max_depth
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='mae', verbose = False)

    # model predict and evaulate.
    y_pred = model.predict(X_test)
    y_true = y_test
    


    return eva_metric(y_true, y_pred)

def get_data(path):
    df = pd.read_excel(path)
    df2 = df.rename({df.columns[0]: 'dirt', df.columns[-1]: 'cost'}, axis=1)
    df2.pop(df.columns[1])
    df = df2.copy()
    data_cost = df.pop('cost')
    label = df.pop('dirt')
    
    # add time msg
    index = list(range(len(df)))
    df['time'] = pd.DataFrame(index)
    
    return df, label

def lr_curve(results, ylabel, title):
    keys = list(results.keys())
    item = list(results[keys[0]].keys())[0]
    epochs = len(results[keys[0]][item])
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_axis, results["validation_0"][item], label="Train")
    ax.plot(x_axis, results["validation_1"][item], label="Test")
    ax.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def eva_metric(y_true, y_pred):
    dic = {'mse':mean_squared_error,
       'mae': mean_absolute_error,
       'r2_score': r2_score}
    
    r2 = []
    eval_res = {}
    for i in dic.keys():
        score = dic[i](y_true, y_pred)
#         print(i, score)
        eval_res[i] = score
        if i == 'r2_score':
            r2.append(score)
    return eval_res

# X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.33, random_state=42)


def xgb_grid_test(df, label):
    
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.33, random_state=42)
    
    plot = False
    dic = {'obj':['reg:squarederror', 'reg:linear'],
           'booster':['gblinear', 'gbtree', 'dart'],
           'max_depth':[4, 8, 12]}

    best_lis = []
    record_dic = {'obj':[],
                  'booster':[],
                  'max_depth':[],
                  'score':[] }
    max_score = 0
    for a in dic['obj']:
        for b in dic['booster']:
            for c in dic['max_depth']:
                print(a, b, 'max_depth =', c)

                # model training 
                model = xgb_model(200, a, b, c)
                model.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        eval_metric='mae', verbose = False)

                # model predict and evaulate.
                y_pred = model.predict(X_test)
                y_true = y_test
                r2 = eva_metric(y_true, y_pred)['r2_score']

                record_dic['obj'].append(a)
                record_dic['booster'].append(b)
                record_dic['max_depth'].append(c)
                record_dic['score'].append(r2)

                # record best score
                if r2 > max_score:
                    max_score = r2
                    best_lis.append(a+' '+b+' '+'max_depth ='+str(c))

                results = model.evals_result()

                # plot lr curve for mae
                if plot:
                    lr_curve(results, 'xgb linear', 'learning curve of gblinear')
    print(best_lis, max_score)
    return record_dic


def svm_grid_test(df, label):
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.33, random_state=42)

    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, 
                   coef0=1, verbose = True)            # poly 跑很久 沒再動
    lis = []
    dic = {'svr_rbf':svr_rbf,
           'svr_lin':svr_lin} 
    for i in list(dic.keys()):
        print(i)
        model = dic[i]
        y_train = np.array(y_train)
#         y_train = np.squeeze(y_train, axis = -1)
        
        print('train svm with '+i)
        model.fit(X_train, np.array(y_train))
        
        y_true = y_test
        y_pred = model.predict(X_test)
        r2 = eva_metric(y_true, y_pred)
#         print(i, r2)
        lis.append([i, r2['r2_score']])
    return lis