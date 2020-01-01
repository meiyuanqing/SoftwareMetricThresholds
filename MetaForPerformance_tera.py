'''

v1.1,当把做断点回归的因变量换成几率比后，元分析的结果中由于调整阈值的数值偏大，出现阈值为负值情况，该情况不符合实际情况，舍去。
v1:把v0版本上的参数整理出来供调用。


本代码主要实现从metricThresholds.csv文件中读入每个度量的四种阈值，经过下面四个文件调整后的阈值做元分析
    metricThresholds.csv: 主要用到每个项目度量名、LOGIT回归系数的协方差矩阵中cov11、cov12和cov22值，该值为计算在VARLThreshold加上
                        偏移量后新的阈值的主差；为元分析计算服务
    下面四个文件主要用到文件中的rd_bandwidth和times两个变量相乘后的偏移量。
    (1)MaxLATErdrobust.csv: 以VARL为参照，在度量值范围内[min_metric_value,VARL_threshold_value](度量与缺陷正相关)和
                           [VARL_threshold_value,max_metric_value(度量与缺陷负相关)查找LATE估计值最大的cutoff为阈值，
                           并以该阈值为断点的回归结果，但主要是每个断点与VARL的偏移量；
    (2)VARLGreaterAverageGMrdrobust.csv：求解VARL过程中，使用分层10折交叉验证方法，选择其在1折测试集上GM预测性能超过所有平均值以上
                                         那些折的阈值平均值做阈值；
    (3)VARLMaxGMrdrobust.csv： 求解VARL过程中，使用分层10折交叉验证方法，选择其在1折测试集上GM预测性能最大的那一折阈值做阈值

    (4)VARLrdrobust.csv： 求解VARL过程中，使用分层10折交叉验证方法，重复10次的VARL阈值平均值作为阈值；

注意事项：上述四个文件中不能有重复的数据，否则报错,即thresholdAjust = float(thresholdAjustList)，从列表中转实型数会报错，因为列表中多于一个元素。

'''
workingDirectory = "/home/mei/RD/terapromise/"
resultDirectory = "/home/mei/RD/terapromise/thresholds/"
rdrobustDirectory = "/home/mei/RD/terapromise/thresholds/rdrobust/"
metaDirectory = "/home/mei/RD/terapromise/thresholds/metaAnalysis/"


def MetaForPerformance(wd ="/home/mei/RD/terapromise/",
                       rd = "/home/mei/RD/terapromise/thresholds/",
                       rdd = "/home/mei/RD/terapromise/thresholds/rdrobust/",
                       md = "/home/mei/RD/terapromise/thresholds/metaAnalysis/"):
    import os
    import csv
    from scipy.stats import norm      # norm.cdf() the cumulative normal distribution function in Python
    from scipy import stats           # 根据卡方分布计算p值: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    import numpy as np
    import pandas as pd

    workingDirectory = wd
    resultDirectory = rd
    rdrobustDirectory = rdd
    metaDirectory = md

    print(os.getcwd())
    os.chdir(workingDirectory)
    print(os.getcwd())

    # 输入：两个匿名数组，studyEffectSize中存放每个study的effect size，studyVariance存放对应的方差
    # 输出：fixed effect model元分析后的结果，包括
    #      (1) $fixedMean        #固定模型元分析后得到的效应平均值
    #      (2) $fixedStdError    #固定模型元分析的效应平均值对应的标准错
    def fixedEffectMetaAnalysis(effectSize, variance):
        fixedWeight = []
        sum_Wi = 0
        sum_WiYi = 0
        d = {}  # return a dict
        studyNumber = len(variance)
        for i in range(studyNumber):
            if variance[i] == 0:
                continue
            fixedWeight[i] = 1 / variance[i]
            sum_Wi = sum_Wi + fixedWeight[i]
            sum_WiYi = sum_WiYi + effectSize[i] * fixedWeight[i]
        fixedMean = sum_WiYi / sum_Wi                       # 固定模型元分析后得到的效应平均值
        fixedStdError = (1 / sum_Wi) ** 0.5                 # 固定模型元分析的效应平均值对应的标准错
        d['fixedMean'] = fixedMean
        d['fixedStdError'] = fixedStdError
        return d

    # 输入：两个匿名数组，studyEffectSize中存放每个study的effect size，studyVariance存放对应的方差
    # 输出：random effect model元分析后的结果，包括
    #      (1) $randomMean        #随机模型元分析后得到的效应平均值
    #      (2) $randomStdError    #随机模型元分析的效应平均值对应的标准错

    def randomEffectMetaAnalysis(effectSize, variance):

        sum_Wi = 0
        sum_WiWi = 0
        sum_WiYi = 0           # Sum(Wi*Yi), where i ranges from 1 to k, where k is the number of studies
        sum_WiYiYi = 0         # Sum(Wi*Yi*Yi), where i ranges from 1 to k, where k is the number of studies

        sum_Wistar = 0
        sum_WistarYi = 0
        d = {}                 # return a dict

        studyNumber = len(variance)
        fixedWeight = [0 for i in range(studyNumber)]  # 固定模型对应的权值
        randomWeight = [0 for i in range(studyNumber)]      # 随机模型对应的权值

        for i in range(studyNumber):
            if variance[i] == 0:
                continue
            fixedWeight[i] = 1 / variance[i]
            # if fixedWeight[i] == 0:
            #     continue
            sum_Wi = sum_Wi + fixedWeight[i]
            # if sum_Wi == 0:
            #     continue
            sum_WiWi = sum_WiWi + fixedWeight[i] * fixedWeight[i]
            sum_WiYi = sum_WiYi + effectSize[i] * fixedWeight[i]
            sum_WiYiYi = sum_WiYiYi + fixedWeight[i] * effectSize[i] * effectSize[i]

        # print("the studyNumber is ", studyNumber)
        # print("the sum sum_Wi is ", sum_Wi)

        Q = sum_WiYiYi - sum_WiYi * sum_WiYi / sum_Wi
        df = studyNumber - 1
        C = sum_Wi - sum_WiWi / sum_Wi

        # 当元分析过程中只有一个study研究时，没有研究间效应，故研究间的方差为零
        if studyNumber == 1:
            T2 = 0
        else:
            T2 = (Q - df) / C                 # sample estimate of tau square
        # print("the tau square value is ", T2)
        # print("the len(variance) value is ", studyNumber)
        if T2 < 0:
            T2 = (- 1) * T2             # 20190719，若T2小于，取相反数

        for i in range(studyNumber):
            randomWeight[i] = 1 / (variance[i] + T2)               # randomWeight 随机模型对应的权值

        for i in range(studyNumber):
            sum_Wistar = sum_Wistar + randomWeight[i]
            sum_WistarYi = sum_WistarYi + randomWeight[i] * effectSize[i]

        randomMean = sum_WistarYi / sum_Wistar                      # 随机模型元分析后得到的效应平均值
        randomStdError = (1 / sum_Wistar) ** 0.5                    # 随机模型元分析的效应平均值对应的标准错
        # 当元分析过程中只有一个study研究时，没有研究间异质性，故异质性为零
        if studyNumber == 1:
            I2 = 0
        else:
            I2 = ((Q - df) / Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,
                                  # the proportion of the observed variance reflects real differences in effect size
        # if I2 < 0:
        #     I2 = 0

        pValue_Q = 1.0 - stats.chi2.cdf(Q, df)       # pValue_Q = 1.0 - stats.chi2.cdf(chisquare, freedom_degree)

        d["C"] = C
        d["mean"] = randomMean
        d["stdError"] = randomStdError
        d["LL_CI"] = randomMean - 1.96 * randomStdError     # The 95% lower limits for the summary effect
        d["UL_CI"] = randomMean + 1.96 * randomStdError     # The 95% upper limits for the summary effect
        d["ZValue"] = randomMean / randomStdError      # a Z-value to test the null hypothesis that the mean effect is zero
        d["pValue_Z"] = 2 * (1 - norm.cdf(randomMean / randomStdError))   # norm.cdf() 返回标准正态累积分布函数值
        d["Q"] = Q
        d["df"] = df
        d["pValue_Q"] = pValue_Q
        d["I2"] = I2
        d["tau"] = T2 ** 0.5
        d["LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)   # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        d["UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)   # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
        d["LL_tdPred"] = randomMean - stats.t.ppf(0.975, df) * ((T2 + randomStdError * randomStdError) ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        d["UL_tdPred"] = randomMean + stats.t.ppf(0.975, df) * ((T2 + randomStdError * randomStdError) ** 0.5)

        fixedMean = sum_WiYi / sum_Wi                       # 固定模型元分析后得到的效应平均值
        fixedStdError = (1 / sum_Wi) ** 0.5                 # 固定模型元分析的效应平均值对应的标准错
        d['fixedMean'] = fixedMean
        d['fixedStdError'] = fixedStdError
        return d

    df = pd.read_csv(resultDirectory + "metricThresholds.csv")    # 读入一个项目
    VARL_df = pd.read_csv(rdrobustDirectory + "VARLrdrobust.csv")  # VARLrdrobust，然后读出每个度量中由断点回归调整的阈值,下同。
    VARLGreaterAverageGM_df = pd.read_csv(rdrobustDirectory + "VARLGreaterAverageGMrdrobust.csv")
    VARLMaxGM_df = pd.read_csv(rdrobustDirectory + "VARLMaxGMrdrobust.csv")
    MaxLATE_df = pd.read_csv(rdrobustDirectory + "MaxLATErdrobust.csv")

    metricNamesData = df["metric"].unique()               # 对fieldnames切片取出所有要处理的度量名称,一共20个
    # 对metricNamesData进行排序，保证最终结果的顺序一致
    metricNamesData = [x for x in metricNamesData if str(x) != 'nan']
    metricNamesData.sort()

    VARL_metricNamesData = VARL_df["metricName"].unique() # 对VARLrdrobust文件中fieldnames切片取出所有要处理的度量,有可能该度量未能执行RD
    VARLGreaterAverageGM_metricNamesData = VARLGreaterAverageGM_df["metricName"].unique() # 对VARLrdrobust文件中fieldnames切片取出所有要处理的度量,有可能该度量未能执行RD
    VARLMaxGM_metricNamesData = VARLMaxGM_df["metricName"].unique() # 对VARLrdrobust文件中fieldnames切片取出所有要处理的度量,有可能该度量未能执行RD
    MaxLATE_metricNamesData = MaxLATE_df["metricName"].unique() # 对VARLrdrobust文件中fieldnames切片取出所有要处理的度量,有可能该度量未能执行RD

    k = 0
    # 进行元分析时，把GM=0 或AUC=0.5或AUC小于0.5的度量剔除，因为这些阈值并不能有效度量
    for metric in metricNamesData:
    # for metric in ["lcom"]:  # 第一个实例
        print("the current metric is ", metric)
        print("the current times column items are \n", VARL_df.loc[:, "times"])

        # (1) 处理 VARLrdrobust.csv 中的阈值
        # 读出断点回归结果 VARLrdrobust.csv 文件中的每个cutoff调整的幅度：rd_bandwidth * times
        if metric in VARL_metricNamesData:
            thresholdAjustList = VARL_df[VARL_df["metricName"] == metric].loc[:, "rd_bandwidth"] * VARL_df[VARL_df["metricName"] == metric].loc[:, "times"]
            print("the type of thresholdAjust is ", repr(thresholdAjustList))
            thresholdAjust = float(thresholdAjustList)

            print("the current metric rd.bandwidth is ", VARL_df[VARL_df["metricName"] == metric].loc[:, "rd_bandwidth"])
            print("the current metric times is ", VARL_df[VARL_df["metricName"] == metric].loc[:, "times"])
            print("the current metric thresholdAjust is ", float(thresholdAjustList), "  ", repr(float(thresholdAjustList)))
        else:
            thresholdAjust = 0
        #  将上述算出调整幅度后，对metricThresholds.csv文件中每个cutoff进行调整，并计算调整后新的阈值的方差，为元分析服务
        thresholdEffectSizeTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", "VARLThreshold", "corr"]]
        thresholdEffectSizeTotal = thresholdEffectSizeTotal_first[thresholdEffectSizeTotal_first["k-fold"] == "average"].loc[:,
                                       ["GMs_VARL", "VARLThreshold", "corr"]]
        # 计算调整后的阈值，即断点回归过程中找出的断点阈值
        thresholdEffectSize = thresholdEffectSizeTotal[thresholdEffectSizeTotal["GMs_VARL"] > 0].loc[:,
                              "VARLThreshold"] - thresholdAjust
        corrEffectSize = thresholdEffectSizeTotal[thresholdEffectSizeTotal["GMs_VARL"] > 0].loc[:, "corr"]
        thresholdVarianceTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", 'VARL_variance',
                                                                    'B_0', 'cov11', 'cov12', 'cov22', 'corr_std']]
        thresholdVarianceTotal = thresholdVarianceTotal_first[thresholdVarianceTotal_first["k-fold"] == "average"].loc[:,
                                 ["GMs_VARL", "VARL_variance", 'B_0', 'cov11', 'cov12', 'cov22', 'corr_std']]
        B_0 = thresholdVarianceTotal[thresholdVarianceTotal["GMs_VARL"] > 0].loc[:, "B_0"]
        cov11 = thresholdVarianceTotal[thresholdVarianceTotal["GMs_VARL"] > 0].loc[:, "cov11"]
        cov12 = thresholdVarianceTotal[thresholdVarianceTotal["GMs_VARL"] > 0].loc[:, "cov12"]
        cov22 = thresholdVarianceTotal[thresholdVarianceTotal["GMs_VARL"] > 0].loc[:, "cov22"]
        corrVariance = thresholdVarianceTotal[thresholdVarianceTotal["GMs_VARL"] > 0].loc[:, "corr_std"] ** 2
        if thresholdAjust == 0:
            thresholdVariance = thresholdVarianceTotal[thresholdVarianceTotal["GMs_VARL"] > 0].loc[:, "VARL_variance"]
        else:
            threshold_se = ((cov11 + 2 * thresholdEffectSize * cov12 + thresholdEffectSize
                                  * thresholdEffectSize * cov22) ** 0.5) / B_0
            thresholdVariance = threshold_se ** 2
        metaThreshold = pd.DataFrame()
        metaThreshold['EffectSize'] = thresholdEffectSize
        metaThreshold['Variance'] = thresholdVariance

        print("B_0 is ", B_0)
        print("cov11 is ", cov11)
        print("cov12 is ", cov12)
        print("cov22 is ", cov22)

        print("original thresholdEffectSize is ", thresholdEffectSizeTotal[thresholdEffectSizeTotal["GMs_VARL"] > 0].loc[:, "VARLThreshold"])
        print("thresholdAjust is ", thresholdAjust)
        print("thresholdEffectSize is ", thresholdEffectSize)
        print("thresholdVariance is ", thresholdVariance)
        print("the length of thresholdEffectSize is ", len(thresholdEffectSize))
        print("the length of thresholdEffectSize is ", len(thresholdEffectSize))
        print("the length of thresholdVariance is ", len(thresholdVariance))
        try:
            resultMetaAnalysis = randomEffectMetaAnalysis(
                np.array(metaThreshold[metaThreshold["EffectSize"] > 0].loc[:, "EffectSize"]),
                np.array(metaThreshold[metaThreshold["EffectSize"] > 0].loc[:, "Variance"]))

        # resultMetaAnalysis = randomEffectMetaAnalysis(np.array(thresholdEffectSize), np.array(thresholdVariance))

            print("the meta-analysis result of metric of ", metric, "'s threshold is ", resultMetaAnalysis["mean"])
            print("the meta-analysis result of metric of ", metric, "'s stdError is ", resultMetaAnalysis["stdError"])
            with open(metaDirectory + "VARL_metaThresholds.csv", 'a+', encoding="utf-8", newline='') as VARL_f:
                writer_VARL_f = csv.writer(VARL_f)
                if os.path.getsize(metaDirectory + "VARL_metaThresholds.csv") == 0:
                    writer_VARL_f.writerow(["metric", "metaThresholds", "stdError"])
                writer_VARL_f.writerow([metric, resultMetaAnalysis["mean"], resultMetaAnalysis["stdError"]])

        except Exception as err1:
            print(err1)

        try:
            corrresultMetaAnalysis = randomEffectMetaAnalysis(np.array(corrEffectSize), np.array(corrVariance))

            with open(metaDirectory + "corr_meta.csv", 'a+', encoding="utf-8", newline='') as corr_f:
                writer_corr_f = csv.writer(corr_f)
                if os.path.getsize(metaDirectory + "corr_meta.csv") == 0:
                    writer_corr_f.writerow(["metric", "metaCorrs", "stdError"])
                writer_corr_f.writerow([metric, corrresultMetaAnalysis["mean"], corrresultMetaAnalysis["stdError"]])

        except Exception as errcorr:
            print(errcorr)

        # (2) 处理 VARLMaxGMrdrobust.csv 中的阈值    VARLMaxGM_df
        if metric in VARLMaxGM_metricNamesData:
            VARLMaxGM_T_AjustList = VARLMaxGM_df[VARLMaxGM_df["metricName"] == metric].loc[:, "rd_bandwidth"] * VARLMaxGM_df[VARLMaxGM_df["metricName"] == metric].loc[:, "times"]
            VARLMaxGM_T_Ajust = float(VARLMaxGM_T_AjustList)
            print("the current metric rd.bandwidth is ", VARLMaxGM_df[VARLMaxGM_df["metricName"] == metric].loc[:, "rd_bandwidth"])
            print("the current metric times is ", VARLMaxGM_df[VARLMaxGM_df["metricName"] == metric].loc[:, "times"])
            print("the current metric VARLMaxGM_T_Ajust is ", float(VARLMaxGM_T_Ajust), "  ", repr(float(VARLMaxGM_T_Ajust)))
        else:
            VARLMaxGM_T_Ajust = 0

        VARLMaxGM_T_EffectSizeTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", "VARLThreshold"]]
        VARLMaxGM_T_EffectSizeTotal = VARLMaxGM_T_EffectSizeTotal_first[VARLMaxGM_T_EffectSizeTotal_first["k-fold"] == "average"].loc[:,
                                       ["GMs_VARL", "VARLThreshold"]]
        # 计算调整后的阈值，即断点回归过程中找出的断点阈值
        VARLMaxGM_T_EffectSize = VARLMaxGM_T_EffectSizeTotal[VARLMaxGM_T_EffectSizeTotal["GMs_VARL"] > 0].loc[:,
                              "VARLThreshold"] - VARLMaxGM_T_Ajust
        VARLMaxGM_T_VarianceTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", 'VARL_variance',
                                                                    'B_0', 'cov11', 'cov12', 'cov22']]
        VARLMaxGM_T_VarianceTotal = VARLMaxGM_T_VarianceTotal_first[VARLMaxGM_T_VarianceTotal_first["k-fold"] == "average"].loc[:,
                                 ["GMs_VARL", "VARL_variance", 'B_0', 'cov11', 'cov12', 'cov22']]
        B_0 = VARLMaxGM_T_VarianceTotal[VARLMaxGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "B_0"]
        cov11 = VARLMaxGM_T_VarianceTotal[VARLMaxGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov11"]
        cov12 = VARLMaxGM_T_VarianceTotal[VARLMaxGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov12"]
        cov22 = VARLMaxGM_T_VarianceTotal[VARLMaxGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov22"]
        if VARLMaxGM_T_Ajust == 0:
            VARLMaxGM_T_Variance = VARLMaxGM_T_VarianceTotal[VARLMaxGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "VARL_variance"]
        else:
            VARLMaxGM_T_se = ((cov11 + 2 * VARLMaxGM_T_EffectSize * cov12 + VARLMaxGM_T_EffectSize
                                    * VARLMaxGM_T_EffectSize * cov22) ** 0.5) / B_0
            VARLMaxGM_T_Variance = VARLMaxGM_T_se ** 2

        print(VARLMaxGM_T_EffectSize)
        print(VARLMaxGM_T_Variance)

        print("the length of VARLMaxGM_T_EffectSize is ", len(VARLMaxGM_T_EffectSize))
        print("the length of VARLMaxGM_T_Variance is ", len(VARLMaxGM_T_Variance))

        VARLMaxGM_metaThreshold = pd.DataFrame()
        VARLMaxGM_metaThreshold['EffectSize'] = VARLMaxGM_T_EffectSize
        VARLMaxGM_metaThreshold['Variance'] = VARLMaxGM_T_Variance

        try:
            result_VARLMaxGM_MetaAnalysis = randomEffectMetaAnalysis(
                np.array(VARLMaxGM_metaThreshold[VARLMaxGM_metaThreshold["EffectSize"] > 0].loc[:, "EffectSize"]),
                np.array(VARLMaxGM_metaThreshold[VARLMaxGM_metaThreshold["EffectSize"] > 0].loc[:, "Variance"]))

            # result_VARLMaxGM_MetaAnalysis = randomEffectMetaAnalysis(np.array(VARLMaxGM_T_EffectSize), np.array(VARLMaxGM_T_Variance))

            print("the meta-analysis result of metric of ", metric ,"'s threshold is ", result_VARLMaxGM_MetaAnalysis["mean"])
            print("the meta-analysis result of metric of ", metric ,"'s stdError is ", result_VARLMaxGM_MetaAnalysis["stdError"])
            with open(metaDirectory + "VARLMaxGM_metaThresholds.csv", 'a+', encoding="utf-8", newline='') as VARLMaxGM_f:
                writer_VARLMaxGM_f = csv.writer(VARLMaxGM_f)
                if os.path.getsize(metaDirectory + "VARLMaxGM_metaThresholds.csv") == 0:
                    writer_VARLMaxGM_f.writerow(["metric", "metaThresholds", "stdError"])
                writer_VARLMaxGM_f.writerow([metric, result_VARLMaxGM_MetaAnalysis["mean"], result_VARLMaxGM_MetaAnalysis["stdError"]])

        except Exception as err2:
            print(err2)

        # (3) 处理 VARLGreaterAverageGMrdrobust.csv 中的阈值
        if metric in VARLGreaterAverageGM_metricNamesData:
            VARLGreaterAverageGM_T_AjustList = VARLGreaterAverageGM_df[VARLGreaterAverageGM_df["metricName"] == metric].loc[:, "rd_bandwidth"] * VARLGreaterAverageGM_df[VARLGreaterAverageGM_df["metricName"] == metric].loc[:, "times"]
            VARLGreaterAverageGM_T_Ajust = float(VARLGreaterAverageGM_T_AjustList)
        else:
            VARLGreaterAverageGM_T_Ajust = 0

        VARLGreaterAverageGM_T_EffectSizeTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", "VARLThreshold"]]
        VARLGreaterAverageGM_T_EffectSizeTotal = VARLGreaterAverageGM_T_EffectSizeTotal_first[VARLGreaterAverageGM_T_EffectSizeTotal_first["k-fold"] == "average"].loc[:,
                                       ["GMs_VARL", "VARLThreshold"]]
        # 计算调整后的阈值，即断点回归过程中找出的断点阈值
        VARLGreaterAverageGM_T_EffectSize = VARLGreaterAverageGM_T_EffectSizeTotal[VARLGreaterAverageGM_T_EffectSizeTotal["GMs_VARL"] > 0].loc[:,
                              "VARLThreshold"] - VARLGreaterAverageGM_T_Ajust
        VARLGreaterAverageGM_T_VarianceTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", 'VARL_variance',
                                                                    'B_0', 'cov11', 'cov12', 'cov22']]
        VARLGreaterAverageGM_T_VarianceTotal = VARLGreaterAverageGM_T_VarianceTotal_first[VARLGreaterAverageGM_T_VarianceTotal_first["k-fold"] == "average"].loc[:,
                                 ["GMs_VARL", "VARL_variance", 'B_0', 'cov11', 'cov12', 'cov22']]
        B_0 = VARLGreaterAverageGM_T_VarianceTotal[VARLGreaterAverageGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "B_0"]
        cov11 = VARLGreaterAverageGM_T_VarianceTotal[VARLGreaterAverageGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov11"]
        cov12 = VARLGreaterAverageGM_T_VarianceTotal[VARLGreaterAverageGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov12"]
        cov22 = VARLGreaterAverageGM_T_VarianceTotal[VARLGreaterAverageGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov22"]
        if VARLGreaterAverageGM_T_Ajust == 0:
            VARLGreaterAverageGM_T_Variance = VARLGreaterAverageGM_T_VarianceTotal[VARLGreaterAverageGM_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "VARL_variance"]
        else:
            VARLGreaterAverageGM_T_se = ((cov11 + 2 * VARLGreaterAverageGM_T_EffectSize * cov12 + VARLGreaterAverageGM_T_EffectSize
                                    * VARLGreaterAverageGM_T_EffectSize * cov22) ** 0.5) / B_0
            VARLGreaterAverageGM_T_Variance = VARLGreaterAverageGM_T_se ** 2

        VARLGreaterAverageGM_metaThreshold = pd.DataFrame()
        VARLGreaterAverageGM_metaThreshold['EffectSize'] = VARLGreaterAverageGM_T_EffectSize
        VARLGreaterAverageGM_metaThreshold['Variance'] = VARLGreaterAverageGM_T_Variance

        try:
            result_VARLGreaterAverageGM_MetaAnalysis = randomEffectMetaAnalysis(
                np.array(VARLGreaterAverageGM_metaThreshold[VARLGreaterAverageGM_metaThreshold["EffectSize"] > 0].loc[:, "EffectSize"]),
                np.array(VARLGreaterAverageGM_metaThreshold[VARLGreaterAverageGM_metaThreshold["EffectSize"] > 0].loc[:, "Variance"]))

        # result_VARLGreaterAverageGM_MetaAnalysis = randomEffectMetaAnalysis(np.array(VARLGreaterAverageGM_T_EffectSize), np.array(VARLGreaterAverageGM_T_Variance))

            with open(metaDirectory + "VARLGreaterAverageGM_metaThresholds.csv", 'a+', encoding="utf-8", newline='') as VARLGreaterAverageGM_f:
                writer_VARLGreaterAverageGM_f = csv.writer(VARLGreaterAverageGM_f)
                if os.path.getsize(metaDirectory + "VARLGreaterAverageGM_metaThresholds.csv") == 0:
                    writer_VARLGreaterAverageGM_f.writerow(["metric", "metaThresholds", "stdError"])
                writer_VARLGreaterAverageGM_f.writerow([metric, result_VARLGreaterAverageGM_MetaAnalysis["mean"], result_VARLGreaterAverageGM_MetaAnalysis["stdError"]])

        except Exception as err3:
            print(err3)


        # (4) 处理 MaxLATErdrobust.csv 中的阈值
        if metric in MaxLATE_metricNamesData:
            MaxLATE_T_AjustList = MaxLATE_df[MaxLATE_df["metricName"] == metric].loc[:, "rd_bandwidth"] * MaxLATE_df[MaxLATE_df["metricName"] == metric].loc[:, "times"]
            MaxLATE_T_Ajust = float(MaxLATE_T_AjustList)
        else:
            MaxLATE_T_Ajust = 0

        MaxLATE_T_EffectSizeTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", "VARLThreshold"]]
        MaxLATE_T_EffectSizeTotal = MaxLATE_T_EffectSizeTotal_first[MaxLATE_T_EffectSizeTotal_first["k-fold"] == "average"].loc[:,
                                       ["GMs_VARL", "VARLThreshold"]]
        # 计算调整后的阈值，即断点回归过程中找出的断点阈值
        MaxLATE_T_EffectSize = MaxLATE_T_EffectSizeTotal[MaxLATE_T_EffectSizeTotal["GMs_VARL"] > 0].loc[:,
                              "VARLThreshold"] - MaxLATE_T_Ajust
        MaxLATE_T_VarianceTotal_first = df[df["metric"] == metric].loc[:, ["k-fold", "GMs_VARL", 'VARL_variance',
                                                                    'B_0', 'cov11', 'cov12', 'cov22']]
        MaxLATE_T_VarianceTotal = MaxLATE_T_VarianceTotal_first[MaxLATE_T_VarianceTotal_first["k-fold"] == "average"].loc[:,
                                 ["GMs_VARL", "VARL_variance", 'B_0', 'cov11', 'cov12', 'cov22']]
        B_0 = MaxLATE_T_VarianceTotal[MaxLATE_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "B_0"]
        cov11 = MaxLATE_T_VarianceTotal[MaxLATE_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov11"]
        cov12 = MaxLATE_T_VarianceTotal[MaxLATE_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov12"]
        cov22 = MaxLATE_T_VarianceTotal[MaxLATE_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "cov22"]
        if MaxLATE_T_Ajust == 0:
            MaxLATE_T_Variance = MaxLATE_T_VarianceTotal[MaxLATE_T_VarianceTotal["GMs_VARL"] > 0].loc[:, "VARL_variance"]
        else:
            MaxLATE_T_se = ((cov11 + 2 * MaxLATE_T_EffectSize * cov12 + MaxLATE_T_EffectSize
                                    * MaxLATE_T_EffectSize * cov22) ** 0.5) / B_0
            MaxLATE_T_Variance = MaxLATE_T_se ** 2

        MaxLATE_metaThreshold = pd.DataFrame()
        MaxLATE_metaThreshold['EffectSize'] = MaxLATE_T_EffectSize
        MaxLATE_metaThreshold['Variance'] = MaxLATE_T_Variance

        try:
            result_MaxLATE_MetaAnalysis = randomEffectMetaAnalysis(
               np.array(MaxLATE_metaThreshold[MaxLATE_metaThreshold["EffectSize"] > 0].loc[:, "EffectSize"]),
               np.array(MaxLATE_metaThreshold[MaxLATE_metaThreshold["EffectSize"] > 0].loc[:, "Variance"]))

        # result_MaxLATE_MetaAnalysis = randomEffectMetaAnalysis(np.array(MaxLATE_T_EffectSize), np.array(MaxLATE_T_Variance))

            with open(metaDirectory + "MaxLATE_metaThresholds.csv", 'a+', encoding="utf-8", newline='') as MaxLATE_f:
                writer_MaxLATE_f = csv.writer(MaxLATE_f)
                if os.path.getsize(metaDirectory + "MaxLATE_metaThresholds.csv") == 0:
                    writer_MaxLATE_f.writerow(["metric", "metaThresholds", "stdError"])
                writer_MaxLATE_f.writerow([metric, result_MaxLATE_MetaAnalysis["mean"], result_MaxLATE_MetaAnalysis["stdError"]])

        except Exception as err4:
            print(err4)

        k += 1
        # if k == 1:
        #     break

if __name__ == '__main__':
    MetaForPerformance()
    pass