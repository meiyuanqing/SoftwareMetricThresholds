# !/usr/bin python3.6
# encoding: utf-8
# @author = 'Yuanqing Mei'
# @contact: dg1533019@smail.nju.edu.cn
# @file: ManualDown.py
# @time: 2020/7/6 下午8:55

'''

this script will generate the performance indicators of ManualDown model.

参考文献：
[1]  zhou et al. how far...
'''

# 参数说明：
#    (1) wd： 用于存放被训练的项目路径，默认值为"/home/mei/RD/terapromise/scripts/ManualDown/data/"；
#    (2) rd： 用于存放模型预测性能指标，默认值是"/home/mei/RD/terapromise/scripts/ManualDown/thresholds/"。
#    (3) trainl:  训练集的文件列表，即wd路径下文件名。默认值是："ListTraining.txt"

def manualDown(wd="/home/mei/RD/terapromise/scripts/ManualDown/data/",
                rd="/home/mei/RD/terapromise/scripts/ManualDown/thresholds/",
                trainl="ListTraining.txt"):
    import os
    import csv
    import numpy as np
    import math
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix

    workingDirectory = wd
    resultDirectory = rd

    print(os.getcwd)
    os.chdir(workingDirectory)
    print(os.getcwd())

    with open(resultDirectory + trainl) as l:
        lines = l.readlines()

    for line in lines:
        file = line.replace("\n", "")

        # 分别处理每一个项目:f1取出要被处理的项目；f2：用于存储每一个项目的预测性能指标信息；
        # f2用csv.writer写数据时没有newline参数，会多出一空行
        with open(workingDirectory + file, 'r', encoding="ISO-8859-1") as f1, \
                open(resultDirectory + "metrics_" + "performance.csv", 'a+', encoding="utf-8", newline='') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
            # 对每个项目文件查看一下，把属于metric度量的字段整理到metricData
            metricData = fieldnames[3:24]  # 对fieldnames切片取出所有要处理的度量,一共20个
            print("the metricData is ", metricData)
            # 先写入columns_name
            if os.path.getsize(resultDirectory + "metrics_" + "performance.csv") == 0:
                writer.writerow(["fileName", "metric", "TP", "TN", "FP", "FN", "recall", "precision", "g-mean", "AUC",
                                 "ER", "numberOfSLOC", "slocLocation", "slocThreshold"])

            df = pd.read_csv(file)

            # manualDown model只需要loc和bug度量数据
            for metric in metricData:

                print("the current file is ", file)
                print("the current metric is ", metric)
                if metric == "bug":
                    continue
                # if not (metric == "bug" or metric == "loc"):
                #     continue
                # 若度量中存在undef或undefined数据，由于使得每个度量值的个数不同，故舍去该度量的值
                undef = 0
                undefined = 0
                for x in df[metric]:
                    if x == 'undef':
                        undef = 1
                    if x == 'undefined':
                        undefined = 1
                if undef:
                    continue
                if undefined:
                    continue

                # 由于bug中存储的是缺陷个数，转化为二进制形式存储;
                # 此次可调整lambda中条件参数，若x>2,则可预测bug为3个以上的阈值，其他类推
                df['bugBinary'] = df.bug.apply(lambda x: 1 if x > 0 else 0)

                # 根据ManualDown(50%),预测从大到小，前50%(向上取整)为defective。
                # 新增一列，按从大到小的loc值给每一行一个序号，然后取序号的前一半为defective,若个数为奇数，中位数位置上的也是defective
                # 思路是按规模从大到小取出第50%位置上的LOC值，再根据此值来预测
                sortedSLOC = sorted(df['loc'], reverse=True)
                print("the sortedSLOC is ", sortedSLOC)
                print("the max of sortedSLOC is ", max(sortedSLOC))
                print("the type of sortedSLOC is ", type(sortedSLOC))
                # 定义df["predictBinary"]初始值为空
                df["predictBinary"] = ""
                # loc度量前50%的df["predictBinary"]为1
                for i in range(math.ceil(len(sortedSLOC)/2)):
                    slocLocation = i
                    slocThreshold = max(sortedSLOC)
                    for j in range(len(df['loc'])):
                        if df.loc[j, "loc"] == max(sortedSLOC):
                            if df.loc[j,"predictBinary"] == "":
                                df.loc[j, "predictBinary"] = 1
                                break
                    sortedSLOC.remove(max(sortedSLOC))
                    # print("the remove max of sortedSLOC is ", sortedSLOC)
                # loc度量后50%的df["predictBinary"]为0
                for k in range(len(df['loc'])):
                    if df.loc[k, "predictBinary"] == "":
                        df.loc[k, "predictBinary"] = 0
                print("the slocLocation is ", slocLocation)
                print("the slocThreshold is ", slocThreshold)
                # print("the predictBinary is ", df.loc[:, ['loc', 'bug', 'predictBinary']])


                # 计算GM性能指标，其公式为GM=(TPR*TNR)**0.5
                # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
                confusionMatrix = confusion_matrix(df["bugBinary"], df['predictBinary'], labels=[0, 1])
                print("confusionMatrix is:\n", confusionMatrix)
                # 计算混淆矩阵时有些情况只能算出一个TN或TP的
                tn, fp, fn, tp = confusionMatrix.ravel()
                # 当tp且tn为零时，GM值为零，因为GM=(TPR*TNR)**0.5
                if tp == 0 and tn == 0:
                    GM = 0
                elif tp == 0 and tn != 0:
                    GM = (tn / (tn + fp)) ** 0.5
                elif tp != 0 and tn == 0:
                    GM = (tp / (tp + fn)) ** 0.5
                else:
                    GM = ((tp / (tp + fn)) * (tn / (tn + fp))) ** 0.5

                # 计算AUC性能指标
                fpr, tpr, thresholds = roc_curve(df["bugBinary"], df['predictBinary'])
                AUC = auc(fpr, tpr)

                # 计算ER指标
                df["si*pi"] = df["loc"] * df['predictBinary']
                Effort_m = df["si*pi"].sum() / df["loc"].sum()
                df["fi*pi"] = df["bug"] * df['predictBinary']
                Effort_random = df["fi*pi"].sum() / df["bug"].sum()
                ER_m = (Effort_random - Effort_m) / Effort_random

                if metric == "loc":
                    writer.writerow([file, metric, tp, tn, fp, fn, tp/(tp + fn), tp/(tp + fp), GM, AUC, ER_m,
                                     df['loc'].count(), slocLocation + 1, slocThreshold])

                # 输出每一个度量值的预测后tp,tn下度量值集合，为Venn图准备数据
                with open(resultDirectory + metric + "_ManualDown_venn.csv", 'a+', encoding="utf-8", newline='') as f3:
                    writer_f3 = csv.writer(f3)

                    # 当文件大小为零，即刚创建时，则输出标题
                    if os.path.getsize(resultDirectory + metric + "_ManualDown_venn.csv") == 0:
                        writer_f3.writerow(["fileName", metric, "bug", "bugBinary", "loc", "slocThreshold",
                                            "predictBinary", "TP", "TN"])
                    # 通过循环输出数据
                    for i in range(len(df)):
                        tp_value = ""
                        tn_value = ""
                        if df.loc[i, "predictBinary"] == df.loc[i, "bugBinary"]:

                            if df.loc[i, "predictBinary"] == 1:
                                tp_value = df.loc[i, metric]

                            if df.loc[i, "predictBinary"] == 0:
                                tn_value = df.loc[i, metric]


                        writer_f3.writerow([file, df.loc[i, metric], df.loc[i, "bug"], df.loc[i, "bugBinary"],
                                            df.loc[i, "loc"], slocThreshold, df.loc[i, "predictBinary"],
                                            tp_value, tn_value])

    print("This python file is ManualDown.py!")

if __name__ == '__main__':
    manualDown()
    print("This is end of manualDown model!")
    pass