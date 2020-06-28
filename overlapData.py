# !/usr/bin python3.6
# encoding: utf-8
# @author = 'Yuanqing Mei'
# @contact: dg1533019@smail.nju.edu.cn
# @file: overlapData.py
# @time: 2020/6/27 下午2:56

'''
this script will present characteristic of the overlapping range between defective and non defective modules.
以重叠区域占度量值域的百分比为解释变量； 预测性能为被解释变量，进行最小二乘回归
'''

import os
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

wd = "/home/mei/RD/terapromise/"
rd = "/home/mei/RD/terapromise/thresholds/overlapData/"
trainl= "ListTraining.txt"

workingDirectory = wd
resultDirectory = rd

print(os.getcwd)
os.chdir(workingDirectory)
print(os.getcwd())

with open(resultDirectory + trainl) as l:
    lines = l.readlines()

print(lines)
# draw a box plot comparing defective and non-defective module
metricData = []

metricDict = {}
defectDict = {}
nonDefectDict = {}

metricDictOverlapping = {}
defectDictOverlapping = {}
nonDefectDictOverlapping = {}

gmdf = pd.read_csv(rd + "gm.csv")
print(gmdf)
# for line in lines:
#     file = line.replace("\n", "")
#     print(gmdf[gmdf["project.GM"] == file].loc[:, "amc"])
#     temp = gmdf[gmdf["project.GM"] == file].loc[:, "amc"]
#     print(float(temp))
#     # print(temp.iloc[0, 0])

for line in lines:
    file = line.replace("\n", "")

    # 分别处理文件中的每一个项目:f1取出要被处理的项目；f2：用于存储每一个项目的重叠部分数据信息；
    # f2用csv.writerp写数据时没有newline参数，会多出一空行
    with open(workingDirectory + file, 'r', encoding="ISO-8859-1") as f1, \
            open(resultDirectory + "overlappingData.csv", 'a+', encoding="utf-8", newline='') as f2:
        reader = csv.reader(f1)
        writer = csv.writer(f2)
        fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
        # 对每个项目文件查看一下，把属于metric度量的字段整理到metricData
        metricData = fieldnames[3:23]  # 对fieldnames切片取出所有要处理的度量,一共20个
        metricData.sort()

        # 先写入columns_name
        if os.path.getsize(resultDirectory + "overlappingData.csv") == 0:
            writer.writerow(["fileName", "metric", "min_DefectMetric", "max_DefectMetric", "min_nonDefectMetric",
                             "max_nonDefectMetric", "min_overlapping", "max_overlapping", "overlapping",
                             "rangeMetric", "rangeProportion",
                             "min_Metric", "max_Metric", "countOverlapping_Defective", "countOverlapping_nonDefective",
                             "countOverlapping", "countMetric", "countProportion", "gm"
                             ])

        df = pd.read_csv(file)

        if not os.path.exists(rd + '/Boxplot/92projects/' + file):
            os.makedirs(rd + '/Boxplot/92projects/' + file)

        for metric in metricData:

            print("the current file is ", file)
            print("the current metric is ", metric)
            if metric == "bug":
                continue

            gm = float(gmdf[gmdf["project.GM"] == file].loc[:, metric])

            # 第一次用字典判断是否存在，不存在创建一个空字典
            if metric not in metricDict:
                metricDict[metric] = []

            if metric not in defectDict:
                defectDict[metric] = []

            if metric not in nonDefectDict:
                nonDefectDict[metric] = []

            # # 第一次用字典判断是否存在，不存在创建一个空字典
            if metric not in metricDictOverlapping:
                metricDictOverlapping[metric] = []

            if metric not in defectDictOverlapping:
                defectDictOverlapping[metric] = []

            if metric not in nonDefectDictOverlapping:
                nonDefectDictOverlapping[metric] = []

            # 此次可调整lambda中条件参数，若x>2,则可预测bug为3个以上的阈值，其他类推
            df['bugBinary'] = df.bug.apply(lambda x: 1 if x > 0 else 0)
            # print("metric value of the defective module\n", df[df['bugBinary'] == 1].loc[:, metric])

            #compute all value of metric
            max_Metric = np.max(df[metric])
            min_Metric = np.min(df[metric])
            mean_Metric = np.mean(df[metric])
            std_Metric = np.std(df[metric])
            median_Metric = np.median(df[metric])
            fisrtQuantile_Metric = np.percentile(df[metric], 25)
            thirdQuantile_Metric = np.percentile(df[metric], 75)
            print("the min all value\t", min_Metric, "the 1Q all value\t", fisrtQuantile_Metric,
                  "the median all value\t", median_Metric, "the 3Q all value\t", thirdQuantile_Metric,
                  "the max all value\t", max_Metric, "the mean all value\t", mean_Metric,
                  "the std all value\t", std_Metric)

            # compute all value of metric of defective module
            max_DefectMetric = np.max(df[df['bugBinary'] == 1].loc[:, metric])
            min_DefectMetric = np.min(df[df['bugBinary'] == 1].loc[:, metric])
            mean_DefectMetric = np.mean(df[df['bugBinary'] == 1].loc[:, metric])
            std_DefectMetric = np.std(df[df['bugBinary'] == 1].loc[:, metric])
            median_DefectMetric = np.median(df[df['bugBinary'] == 1].loc[:, metric])
            fisrtQuantile_DefectMetric = np.percentile(df[df['bugBinary'] == 1].loc[:, metric], 25)
            thirdQuantile_DefectMetric = np.percentile(df[df['bugBinary'] == 1].loc[:, metric], 75)

            print("the min Defect value\t", min_DefectMetric, "the 1Q Defect value\t", fisrtQuantile_DefectMetric,
                  "the median Defect value\t", median_DefectMetric, "the 3Q Defect value\t", thirdQuantile_DefectMetric,
                  "the max Defect value\t", max_DefectMetric, "the mean Defect value\t", mean_DefectMetric,
                  "the std Defect value\t", std_DefectMetric)

            # compute all value of metric of non defective module
            max_nonDefectMetric = np.max(df[df['bugBinary'] == 0].loc[:, metric])
            min_nonDefectMetric = np.min(df[df['bugBinary'] == 0].loc[:, metric])
            mean_nonDefectMetric = np.mean(df[df['bugBinary'] == 0].loc[:, metric])
            std_nonDefectMetric = np.std(df[df['bugBinary'] == 0].loc[:, metric])
            median_nonDefectMetric = np.median(df[df['bugBinary'] == 0].loc[:, metric])
            fisrtQuantile_nonDefectMetric = np.percentile(df[df['bugBinary'] == 0].loc[:, metric], 25)
            thirdQuantile_nonDefectMetric = np.percentile(df[df['bugBinary'] == 0].loc[:, metric], 75)

            print("the min nonDefect value\t", min_nonDefectMetric, "\t",
                  "the 1Q nonDefect value\t", fisrtQuantile_nonDefectMetric, "\t",
                  "the median nonDefect value\t", median_nonDefectMetric, "\t",
                  "the 3Q nonDefect value\t", thirdQuantile_nonDefectMetric, "\t",
                  "the max nonDefect value\t", max_nonDefectMetric, "\t",
                  "the mean nonDefect value\t", mean_nonDefectMetric,"\t",
                  "the std nonDefect value\t", std_nonDefectMetric,
                  )

            #重叠区域值域占总值域的比例，若是离散型变量，则需要
            rangeMetric = max_Metric - min_Metric
            if (min_nonDefectMetric > max_DefectMetric):
                overlapping = 0
                rangeProportion = 0
                print("there is no overlapping")
            elif (min_DefectMetric > max_nonDefectMetric):
                overlapping = 0
                rangeProportion = 0
                print("there is no overlapping")
            else:
                # min_overlapping = np.max([min_nonDefectMetric, min_DefectMetric], 0) #np.max()中只能接一给数组，第二参数是维数
                min_overlapping = np.max([min_nonDefectMetric, min_DefectMetric]) #np.max()中只能接一给数组，第二参数是维数
                print("the repr of max_nonDefectMetric is ", type(max_nonDefectMetric))
                print("the repr of max_DefectMetric is ", type(max_DefectMetric))
                max_overlapping = np.min([max_DefectMetric, max_nonDefectMetric])
                overlapping = max_overlapping - min_overlapping
                if rangeMetric == 0:
                    rangeProportion = 0
                else:
                    rangeProportion = overlapping / rangeMetric
                print("the min overlapping is: ", min_overlapping,
                      "the max overlapping is: ", max_overlapping,
                      "the overlapping is: ", max_overlapping - min_overlapping,
                      )

            #重叠区域中模块数占总模块的比例
            dfRange = pd.DataFrame()
            dfRange[metric] = df[df[metric] >= min_overlapping].loc[:, metric]
            dfRange['bugBinary'] = df[df[metric] >= min_overlapping].loc[:, 'bugBinary']
            dfRange = dfRange[dfRange[metric] <= max_overlapping]

            countOverlapping = dfRange[metric].shape[0]
            countMetric = df[metric].shape[0]
            countProportion = dfRange[metric].shape[0] / df[metric].shape[0]

            countOverlapping_Defective = dfRange[dfRange["bugBinary"] == 1].shape[0]
            countOverlapping_nonDefective = dfRange[dfRange["bugBinary"] == 0].shape[0]


            print("the proportion of overlapping is ", countProportion)
            print("the number of overlapping is ", dfRange[metric].shape[0])
            print("the number of metric is ", df[metric].shape[0])

            writer.writerow([file, metric, min_DefectMetric, max_DefectMetric, min_nonDefectMetric, max_nonDefectMetric,
                             min_overlapping, max_overlapping, overlapping, rangeMetric, rangeProportion,
                             min_Metric, max_Metric, countOverlapping_Defective, countOverlapping_nonDefective,
                             countOverlapping, countMetric, countProportion, gm])

            with open(resultDirectory + metric + "_overlappingData.csv", 'a+', encoding="utf-8", newline='') as fmetric:
                writerMetric = csv.writer(fmetric)
                if os.path.getsize(resultDirectory + metric + "_overlappingData.csv") == 0:
                    writerMetric.writerow(
                        ["fileName", "min_DefectMetric", "max_DefectMetric", "min_nonDefectMetric",
                         "max_nonDefectMetric", "min_overlapping", "max_overlapping", "overlapping",
                         "rangeMetric", "rangeProportion",
                         "min_Metric", "max_Metric", "countOverlapping_Defective", "countOverlapping_nonDefective",
                         "countOverlapping", "countMetric", "countProportion", "gm"
                         ])

                writerMetric.writerow([file, min_DefectMetric, max_DefectMetric, min_nonDefectMetric,
                                       max_nonDefectMetric, min_overlapping, max_overlapping, overlapping, rangeMetric,
                                       rangeProportion, min_Metric, max_Metric, countOverlapping_Defective,
                                       countOverlapping_nonDefective,
                                       countOverlapping, countMetric, countProportion, gm])

            # draw a box plot comparing defective and non-defective module
            # np.hstack: horizontal stack,即左右合并;np.vstack: vertical stack,属于一种上下合并,即对括号中的两个整体进行对应操作

            print("the repr of  is ", repr(metricDict[metric]))
            metricDict[metric] = np.hstack((np.array(metricDict[metric]), np.array(df[metric])))
            print("the repr of  is ", repr(metricDict[metric]))
            print("the repr of  is ", repr(defectDict[metric]))
            defectDict[metric] = np.hstack((np.array(defectDict[metric]), np.array(df[df['bugBinary'] == 1].loc[:, metric])))
            print("the repr of  is ", repr(defectDict[metric]))
            print("the repr of  is ", repr(nonDefectDict[metric]))
            nonDefectDict[metric] = np.hstack((np.array(nonDefectDict[metric]), np.array(df[df['bugBinary'] == 0].loc[:, metric])))
            print("the repr of nonDefectDict is ", repr(nonDefectDict[metric]))

            metricDictOverlapping[metric] = np.hstack(
                (np.array(metricDictOverlapping[metric]), np.array(dfRange[metric])))
            defectDictOverlapping[metric] = np.hstack(
                (np.array(defectDictOverlapping[metric]), np.array(dfRange[dfRange['bugBinary'] == 1].loc[:, metric])))
            nonDefectDictOverlapping[metric] = np.hstack(
                (np.array(nonDefectDictOverlapping[metric]), np.array(dfRange[dfRange['bugBinary'] == 0].loc[:, metric])))

            # data = pd.DataFrame({"all values": s1, "defective": s2, "non-defective": s3})
            data = pd.DataFrame({"defective": pd.Series(defectDict[metric]),
                                 "non-defective": pd.Series(nonDefectDict[metric])})

            data.dropna().astype(float).plot.box(title="Box plot of " + metric)
            # df[metric].dropna().astype(float).plot.box(title="Box plot of " + metric)
            plt.ylabel("values of metric")
            plt.xlabel("")  # 我们设置横纵坐标的标题。
            plt.grid(linestyle="--", alpha=0.3)
            plt.savefig(rd + '/Boxplot/92projects/' + file + '/' + metric + '.jpg')
            plt.close()

            # break
    # break

# 执行SM
#  用于存储以重叠区域占度量值域的百分比为解释变量； 预测性能为被解释变量的结果
with open(resultDirectory + "smOverlapping.csv", 'a+', encoding="utf-8", newline='') as smf:
    writerSM = csv.writer(smf)
    for metricSM in metricData:

        dfSM = pd.read_csv(resultDirectory + metricSM + "_overlappingData.csv")
        # 先写入columns_name
        if os.path.getsize(resultDirectory + "smOverlapping.csv") == 0:
            writerSM.writerow(["metric", "params[1]", "pvalues[1]", "tvalues[1]", "rsquared",
                             "rsquared_adj", "fvalue", "f_pvalue",
                             "min_overlapping_total", "max_overlapping_total",
                             "min_total", "max_total",
                             "totalOverlapping_Defective", "totalOverlapping_nonDefective",
                             "totalOverlapping", "totalMetric", "totalProportion"])

        X = dfSM["countProportion"]
        Y = dfSM["gm"]
        X = sm.add_constant(X) # adding a constant

        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X)

        print_model = model.summary()
        print(print_model)

        print("解释变量回归系数", model.params[1])
        print("解释变量回归系数p值", model.pvalues[1])
        print("解释变量回归系数t值", model.tvalues[1])
        print("提取R方", model.rsquared)
        print("提取调整R方", model.rsquared_adj)
        print("提取F-statistic", model.fvalue)
        print("提取F-statistic 的pvalue", model.f_pvalue)

        totalOverlapping = np.sum(dfSM["countOverlapping"])
        totalOverlapping_Defective = np.sum(dfSM["countOverlapping_Defective"])
        totalOverlapping_nonDefective = np.sum(dfSM["countOverlapping_nonDefective"])
        totalMetric = np.sum(dfSM["countMetric"])
        if totalMetric == 0:
            totalProportion = 0
        else:
            totalProportion = totalOverlapping / totalMetric

        min_overlapping_total = np.min(dfSM["min_overlapping"])
        max_overlapping_total = np.max(dfSM["max_overlapping"])

        min_total = np.min(dfSM["min_Metric"])
        max_total = np.max(dfSM["max_Metric"])

        writerSM.writerow([metricSM, model.params[1], model.pvalues[1], model.tvalues[1], model.rsquared,
                         model.rsquared_adj, model.fvalue, model.f_pvalue,
                         min_overlapping_total, max_overlapping_total,
                         min_total, max_total,
                         totalOverlapping_Defective, totalOverlapping_nonDefective,
                         totalOverlapping, totalMetric, totalProportion])


def box_plot_outliers(data_ser, box_scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    val_low = data_ser.quantile(0.25) - iqr
    val_up = data_ser.quantile(0.75) + iqr
    rule_low = (data_ser < val_low)
    rule_up = (data_ser > val_up)
    return (rule_low, rule_up), (val_low, val_up)


# draw a box plot comparing defective and non-defective module
for metricB in metricData:

    if metricB not in metricDict:
        continue

    print("the repr of is ", repr(metricDict[metricB]))
    s1 = pd.Series(metricDict[metricB])
    s2 = pd.Series(defectDict[metricB])
    s3 = pd.Series(nonDefectDict[metricB])

    # data = pd.DataFrame({"all values": s1, "defective": s2, "non-defective": s3})
    data = pd.DataFrame({"defective": s2, "non-defective": s3})

    data.dropna().astype(float).plot.box(title="Box plot of " + metricB)
    # df[metric].dropna().astype(float).plot.box(title="Box plot of " + metric)
    plt.ylabel("values of metric")
    plt.xlabel("")  # 我们设置横纵坐标的标题。
    plt.grid(linestyle="--", alpha=0.3)
    plt.savefig(rd + '/Boxplot/' + metricB + '.jpg')
    plt.close()

# draw a box plot comparing defective and non-defective module only in overlapping area
for metricB in metricData:

    if metricB not in metricDict:
        continue

    print("the repr of is ", repr(metricDictOverlapping[metricB]))
    s1 = pd.Series(metricDictOverlapping[metricB])
    s2 = pd.Series(defectDictOverlapping[metricB])
    s3 = pd.Series(nonDefectDictOverlapping[metricB])

####python封装的异常值处理函数（包括箱线图去除异常值等）
    s2_n = s2.copy()
    # s2_series = s2_n["power"]
    rule, value = box_plot_outliers(s2_n, box_scale=3)
    index = np.arange(s2_n.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    s2_n = s2_n.drop(index)
    s2_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(s2_n.shape[0]))

    s3_n = s3.copy()
    # s2_series = s2_n["power"]
    s3_rule, s3_value = box_plot_outliers(s3_n, box_scale=3)
    s3_index = np.arange(s3_n.shape[0])[s3_rule[0] | s3_rule[1]]
    print("Delete number is: {}".format(len(index)))
    s3_n = s3_n.drop(s3_index)
    s3_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(s3_n.shape[0]))


    # data = pd.DataFrame({"all values": s1, "defective": s2, "non-defective": s3})
    data = pd.DataFrame({"defective": s2_n, "non-defective": s3_n})
    # data = pd.DataFrame({"defective": s2, "non-defective": s3, "defective_3IOR": s2_n, "non-defective_3IQR": s3_n})
    # data = pd.DataFrame({"defective": s2, "non-defective": s3})

    data.dropna().astype(float).plot.box(title="Box plot of " + metricB)
    # df[metric].dropna().astype(float).plot.box(title="Box plot of " + metric)
    plt.ylabel("values of metric")
    plt.xlabel("")  # 我们设置横纵坐标的标题。
    plt.grid(linestyle="--", alpha=0.3)
    plt.savefig(rd + '/Boxplot/overlappedData/' + metricB + '.jpg')
    plt.close()