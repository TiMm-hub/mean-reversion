import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
from sklearn import tree
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

def run(return_arr,sort,sort_cap,median_arr,mar_cap,cap = 100,t0 = 0,t1 = 1200,t2 = 1517 ,lag = 1,bottom_num = 10, mean_reversion = True,use_median =False,fee=0.002):
    return_arr = return_arr
    sort = sort
    sort_cap = sort_cap
    median_arr = median_arr
    mar_cap = mar_cap
    if mean_reversion:
        sort= -1*(sort-650)

    sort_shift_1=shift(sort, [0,1], cval=0)/651
    sort_shift_2=shift(sort, [0,2], cval=0)/651
    sort_shift_3=shift(sort, [0,2], cval=0)/651
    sort_shift_7=shift(sort, [0,7], cval=0)/651
    sort_diff = sort_shift_1-sort_shift_2

    def fetch_data_constraint(bottom_num,cap,lag,start,end):
        #sort_cap_rev= -1*(sort_cap-650)
        arr = []
        arr_return = []
        for j in range(start-lag,end-lag):
            for i in range(np.shape(return_arr)[0]):
                if sort[i][j]<bottom_num:
                    if sort_cap[i][j]<cap:
                        if not np.isnan(mar_cap[i][j]):
                            arr_ij=[sort_shift_1[i][j],sort_shift_2[i][j],sort_shift_3[i][j],sort_shift_7[i][j],sort_diff[i][j],mar_cap[i][j]]
                            arr_ij.append(1 if return_arr[i][j+lag]>=median_arr[j+lag] else 0)
                            arr_return.append(return_arr[i][j+lag])
                            arr.append(arr_ij)
        return arr,arr_return

    def fetch_data_constraint_new(bottom_num,cap,lag,start,end,fee):
        #sort_cap_rev= -1*(sort_cap-650)
        arr = []
        arr_return = []
        for j in range(start-lag,end):
            for i in range(np.shape(return_arr)[0]):
                if (sort_cap[i][j]<cap) :
                    if (sort[i][j]<bottom_num):
                        if not np.isnan(mar_cap[i][j]):
                            arr_ij=[sort_shift_1[i][j],sort_shift_2[i][j],sort_shift_3[i][j],sort_shift_7[i][j],sort_diff[i][j],mar_cap[i][j]]
                            arr_ij.append(1 if return_arr[i][j+lag]>=0.003 else 0)
                            arr_return.append(return_arr[i][j+lag])
                            arr.append(arr_ij)
        return arr,arr_return
    
    if use_median:
        train,return_train = fetch_data_constraint(bottom_num,cap,lag,t0,t1)
        test,return_test = fetch_data_constraint(bottom_num,cap,lag,t1,t2)
        print('median return: train')
        np.unique(np.array(train)[:,6],return_counts=True)
        print('median return: test')
        np.unique(np.array(test)[:,6],return_counts=True)
        x_train = np.array(train)[:,:5]
        y_train = np.array(train)[:,6]
        x_test = np.array(test)[:,:5]
        y_test = np.array(test)[:,6]
        
    else:
        train,return_train=fetch_data_constraint_new(bottom_num,cap,lag,t0,t1,fee)
        test,return_test=fetch_data_constraint_new(bottom_num,cap,lag,t1,t2,fee)
        print('median return: train')
        np.unique(np.array(train)[:,6],return_counts=True)
        print('median return: test')
        np.unique(np.array(test)[:,6],return_counts=True)
        x_train = np.array(train)[:,:5]
        y_train = np.array(train)[:,6]
        x_test = np.array(test)[:,:5]
        y_test = np.array(test)[:,6]
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    acc_array = []
    acc_array.append(accuracy_score(clf.predict(x_test),y_test))
    print('Tree model achieve '+str(accuracy_score(clf.predict(x_test),y_test)))


    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param_bst = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}
    param_bst['nthread'] = 4
    param_bst['eval_metric'] = ['auc','error']
    num_round = 100
    bst = xgb.train(param_bst, dtrain, num_round, evallist,early_stopping_rounds=50)
    ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
    acc_array.append(accuracy_score(np.where(ypred>=0.5,1,0),y_test))
    print('XGBoost achieve '+str(accuracy_score(np.where(ypred>=0.5,1,0),y_test)))

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = ['auc','error']
    lgb_tree = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    ypred = lgb_tree.predict(x_test)
    acc_array.append(accuracy_score(np.where(ypred>=0.5,1,0),y_test))
    print('light_GBM achieve '+str(accuracy_score(np.where(ypred>=0.5,1,0),y_test)))
    
    pnl=[]
    pnl.append(np.cumsum(np.array(return_test)-fee)[-1])
    pnl.append(np.cumsum(np.where(bst.predict(dtest)>0.5,bst.predict(dtest)-0.5,-(0.5-bst.predict(dtest)))*(np.array(return_test)-fee))[-1])
    pnl.append(np.cumsum(np.where(lgb_tree.predict(x_test)>0.5,1-lgb_tree.predict(x_test),-(0.5-bst.predict(dtest)))*(np.array(return_test)-fee))[-1])
    print('baseline：'+str(np.cumsum(np.array(return_test)-fee)[-1]))
    print('XGboost：'+str(np.cumsum(np.where(bst.predict(dtest)>0.5,bst.predict(dtest)-0.5,-(0.5-bst.predict(dtest)))*(np.array(return_test)-fee))[-1]))
    print('light GBM：'+str(np.cumsum(np.where(lgb_tree.predict(x_test)>0.5,1-lgb_tree.predict(x_test),-(0.5-bst.predict(dtest)))*(np.array(return_test)-fee))[-1]))
    
    return acc_array,pnl