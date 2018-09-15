# coding: utf-8
import numpy as np
# Matplotlib for additional customization
from matplotlib import pyplot as plt
# Seaborn for plotting and styling
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model, discriminant_analysis, model_selection

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
    f_c = 0
    for C in Cs:
        model = linear_model.LogisticRegression(C=C)
        # model.set_params(**{'normalize': True})
    
        # 训练集用来训练
        model.fit(X_train, y_train)
        # 测试集用来测试
        score = model.score(X_test, y_test)
        if score > max_score:
            max_score = score
            f_c = C
    print(max_score)
    print(f_c)
    model.set_params(C=f_c)
    predict_res = model.predict(X_predict)
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

def load_data(home):
    X_train = np.loadtxt(home + '/data/train/train.csv', delimiter=",")
    y_train = np.loadtxt(home + '/data/train/trainLabels.csv', delimiter=",")
    return model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=None, stratify=y_train)

def main():
    home = "/Users/zhili/src/zuojie/machinelearning/kaggle/data-science-london-scikit-learn"
    X_train, X_test, y_train, y_test = load_data(home)
    X_predict = np.loadtxt(home + '/data/test/test.csv', delimiter=",")
    # use_linear_regression(home, X_train, X_test, y_train, y_test,X_predict)
    use_logistic_regression(home, X_train, X_test, y_train, y_test,X_predict)

if "__main__" == __name__:
    #print (options)
    main()