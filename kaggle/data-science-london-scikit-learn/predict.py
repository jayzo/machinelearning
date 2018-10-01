# coding: utf-8
import numpy as np
# Matplotlib for additional customization
from matplotlib import pyplot as plt
# Seaborn for plotting and styling
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model, discriminant_analysis, model_selection, metrics
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def use_linear_regression(home, X_train, X_test, y_train, y_test,X_predict):
    
    model = LinearRegression()
    model.set_params(**{'normalize': True})
    
    # 训练集用来训练
    model.fit(X_train, y_train)

    # 测试集用来测试
    test_ret = model.predict(X_test)
    # 利用测试集确定阈值
    max_score=0
    f_gap = 0
    for i in range(1,100):
        pre_ret = []
        gap = float(i/100)
        for ret in test_ret:
            if ret >= gap:
                pre_ret.append(1)
            else:
                pre_ret.append(0)
        score=0
        #print(pre_ret)
        for index in range(0, len(pre_ret)):
            if pre_ret[index] == y_test[index]:
                score += 1
        score = float(score) / len(pre_ret)
        if score > max_score:
            max_score = score
            f_gap = gap
        #print("%.2f: %.2f" %(gap,score))

    print(f_gap)
    print(max_score)
    res = []
    index = 1

    predict_res = model.predict(X_predict)
    for one in predict_res:
        if(one >= f_gap):
            res.append(1)
        else:
            res.append(0)
    f_ret = [
        ['Id','Solution']
    ]
    index = 1
    for one in res:
        data = [index, one]
        f_ret.append(data)
        index += 1
    # print(f_ret)
    np.savetxt(home + '/data/test/pre_res.csv', f_ret, fmt="%s", delimiter=",")

def use_logistic_regression(home, X_train, X_test, y_train, y_test,X_predict):
    Cs = np.logspace(-2,4, num=100)
    max_score = 0
    f_model = None
    for C in Cs:
        model = linear_model.LogisticRegression(C=C)
        # model.set_params(**{'normalize': True})
    
        # 训练集用来训练
        model.fit(X_train, y_train)
        # 测试集用来测试
        score = model.score(X_test, y_test)
        if score > max_score:
            max_score = score
            f_model = model
    print(max_score)
    print(f_model.coef_)
    predict_res = f_model.predict(X_predict)
    # print(predict_res[0:10])
    f_ret = [
        ['Id','Solution']
    ]
    index = 1
    for one in predict_res:
        data = [index, int(one)]
        f_ret.append(data)
        index += 1
    # print(f_ret)
    np.savetxt(home + '/data/test/pre_res.csv', f_ret, fmt="%s", delimiter=",")

def use_gbdt(home, X_train, X_test, y_train, y_test,X_predict):
    gbdt=GradientBoostingClassifier(
        learning_rate=0.03, 
        n_estimators=800, 
        max_features=9, 
        subsample=0.7, 
        random_state=10,
        max_depth=3,
        min_samples_split=6,
        min_samples_leaf=13
    )
    gbdt.fit(X_train,y_train)
    print(gbdt.get_params())
    y_pred = gbdt.predict(X_test)
    y_predprob = gbdt.predict_proba(X_test)[:,1]
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))

    # param_test1 = {'n_estimators':list(range(50,500,50))}
    # gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=2,
    #                                 min_samples_leaf=1,max_depth=3,max_features=None, subsample=1.0,random_state=10), 
    #                     param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
    # gsearch1.fit(X_test,y_test)
    # print(gsearch1.cv_results_,gsearch1.best_params_, gsearch1.best_score_)

    # param_test2 = {'max_depth':list(range(1,14,2)), 'min_samples_split':list(range(2,20,1))}
    # gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, min_samples_leaf=1, 
    #     max_features=None, subsample=1.0, random_state=10), 
    # param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
    # gsearch2.fit(X_test,y_test)
    # print(gsearch2.best_params_, gsearch2.best_score_)

    # param_test3 = {'min_samples_split':list(range(2,20,1)), 'min_samples_leaf':list(range(1,20,1))}
    # gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, 
    #     max_features=None, subsample=1.0, random_state=10,max_depth=3), 
    # param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
    # gsearch3.fit(X_test,y_test)
    # print(gsearch3.best_params_, gsearch3.best_score_)
    # param_test4 = {'max_features':list(range(1,len(X_test[0]),1))}
    # gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(
    #     learning_rate=0.1, 
    #     n_estimators=100, 
    #     subsample=1.0, 
    #     random_state=10,
    #     max_depth=3,
    #     min_samples_split=6,
    #     min_samples_leaf=13), 
    # param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
    # gsearch4.fit(X_test,y_test)
    # print(gsearch4.best_params_, gsearch4.best_score_)

    # param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    # gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(
    #     learning_rate=0.1, 
    #     n_estimators=100, 
    #     subsample=1.0, 
    #     random_state=10,
    #     max_depth=3,
    #     min_samples_split=6,
    #     min_samples_leaf=13,
    #     max_features=9), 
    # param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
    # gsearch5.fit(X_test,y_test)
    # print(gsearch5.best_params_, gsearch5.best_score_)

    predict_res = gbdt.predict(X_predict)
    f_ret = [
        ['Id','Solution']
    ]
    index = 1
    for one in predict_res:
        data = [index, int(one)]
        f_ret.append(data)
        index += 1
    # print(f_ret)
    np.savetxt(home + '/data/test/pre_res.csv', f_ret, fmt="%s", delimiter=",")

    

def load_data(home):
    X_train = np.loadtxt(home + '/data/train/train.csv', delimiter=",")
    y_train = np.loadtxt(home + '/data/train/trainLabels.csv', delimiter=",")
    return model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=None, stratify=y_train)
    # return model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=None)

def main():
    home = "/Users/zhili/src/zuojie/machinelearning/kaggle/data-science-london-scikit-learn"
    X_train, X_test, y_train, y_test = load_data(home)
    X_predict = np.loadtxt(home + '/data/test/test.csv', delimiter=",")
    # use_linear_regression(home, X_train, X_test, y_train, y_test,X_predict)
    # use_logistic_regression(home, X_train, X_test, y_train, y_test,X_predict)
    use_gbdt(home, X_train, X_test, y_train, y_test,X_predict)

if "__main__" == __name__:
    #print (options)
    main()