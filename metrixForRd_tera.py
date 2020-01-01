'''
V3版本:把每个项目LOGIT回归系数标准化，即除以每个项目Y的标准差。
v2:把v1版本上的参数整理出来供调用。
   修复v1版本上第235行左右的 if len(GMs_VARL) == 0: continue代码

input:(1)读入ListTraining.txt文件中各项目名，依次读入workingDirectory目录下各个项目
output:(1)生成metricThresholds.csv文件，文件中主要包括每个项目按要求计算的阈值
       (2)生成各个度量的_logitPredict文件，该文件汇总了各个项目的信息，主要包括后续断点回归所需要的X,C,Y

本代码包为汇总实现断点回归中的第一步，其功能主要有：
    1.读每一个项目文件，然后用分层10折交叉验证的方法，即用9折训练集数据得到LOGIT的回归系数，1折测试集数据得到该9折数据的预测效果；
    2.分别得到三种类型的阈值:
        (1) 10折交叉验证的10次重复后平均值的LOGIT回归系数，并代入训练集的数据到得每个度量值下的发生缺陷的概率；
        (2) 10折中测试集数据预测GM值最大的那一折的LOGIT回归系数，并代入训练集的数据到得每个度量值下的发生缺陷的概率；
        (3) 10折中，GM值大于GM平均值那些折的所有结果的平均值；若10折中只有1折符合要求可以输出，则输出与（1）和（2）相同,
            在这种情况下的LOGIT回归系数，并代入训练集的数据到得每个度量值下的发生缺陷的概率；

参考文献：
[1]  Bender, R. Quantitative risk assessment in epidemiological studies investigating threshold effects.
     Biometrical Journal, 41 (1999), 305-319.（计算VARL的SE（标准误）的参考文献P310）

'''


# 参数说明：
#    (1) wd： 用于存放被训练的项目路径，默认值为"/home/mei/RD/JURECZKO/"；
#    (2) rd： 用于存放为断点回归数据（按各个度量名形成文件 ）以及 metricThresholds.csv用入存入LOGIT回归变量的系数协方差，
#             用于计算调整后的阈值计算其方差，为元分析提供数据,默认值是"/home/mei/RD/JURECZKO/thresholds/"。
#    (3) trainl:  训练集的文件列表，即wd路径下文件名。默认值是："TrainingList.txt"
#    (4)
def metrixForRd(wd="/home/mei/RD/terapromise/",
                rd="/home/mei/RD/terapromise/thresholds/",
                trainl="ListTraining.txt"):
    import os
    import csv
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix

    workingDirectory = wd
    resultDirectory = rd

    print(os.getcwd)
    os.chdir(workingDirectory)
    print(os.getcwd())

    with open(resultDirectory + trainl) as l:
        lines = l.readlines()

    # 分别读入每一个文件，再处理该文件：这里需要两次使用交叉验证：   其中第二次需要stratified 10-fold cv
    # 第一次是把所有的文件进行10折交叉验证，第二次是对第一次分出的9折训练集再次进行10折分层交叉验证取10折平均值的LOGIT回归系数

    for line in lines:
        file = line.replace("\n", "")

        # 分别处理文件中的每一个项目:f1取出要被处理的项目；f2：用于存储每一个项目的阈值信息；
        # f3:用于存储每个项目的logit回归系数，用于后续断点回归的running variable
        # f2没有newline参数，会多出一空行
        with open(workingDirectory + file, 'r', encoding="ISO-8859-1") as f1, \
                open(resultDirectory + "metricThresholds.csv", 'a+', encoding="utf-8", newline='') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
            # 对每个项目文件查看一下，把属于metric度量的字段整理到metricData
            metricData = fieldnames[3:23]  # 对fieldnames切片取出所有要处理的度量,一共20个
            # 先写入columns_name
            if os.path.getsize(resultDirectory + "metricThresholds.csv") == 0:
                writer.writerow(["fileName", "metric", "k-fold", "corr", "corr_std", "B_0", "B_1", "B_0_pValue",
                                 "B_1_pValue", "cov11", "cov12", "cov22", "VARLThreshold", "VARL_variance",
                                 "GMs_VARL", "AUCs_VARL", "AUCs_predict", "B_O_standard", "B_1_standard",
                                 "LATE_max_Threshold", "LATE_max_variance", "numberOfLate",
                                 "LATE_min_Threshold", "LATE_min_variance", "the sequence of LATE"])

            # 读入一个项目
            df = pd.read_csv(file)
            # 依次遍历每一个度量
            for metric in metricData:
                print("the current file is ", file)
                print("the current metric is ", metric)
                if metric == "bug":
                    continue
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
                # 如果该项目数量小于10个的话，10折交叉验证分析不能执行。
                if df[metric].count() < 11:
                    continue

                # 为逻辑回归创建所需的data frame , 由于bug中存储的是缺陷个数，转化为二进制形式存储;
                # 此次可调整lambda中条件参数，若x>2,则可预测bug为3个以上的阈值，其他类推
                df['bugBinary'] = df.bug.apply(lambda x: 1 if x > 0 else 0)
                # 执行分层10折交叉验证，保证测试折中正类占总类比与总体相同（分层保持）
                # sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
                skf = StratifiedKFold(n_splits = 10)
                skf.get_n_splits(df[metric], df["bugBinary"])
                # 存储用缺陷为大于0的模块数占所有模块之比作为基础概率时，计算出的阈值
                VARLThresholds = []
                # 存储用缺陷为大于0的模块数占所有模块之比作为基础概率时，计算出的阈值的方差（标准误的平方）
                VARL_variances = []
                # 用于存储每一次logit回归时系数,此处需要小心有可能B_O是自变量前的系数，而B_1是截距项
                B_O = []
                B_1 = []
                # 用于存储每一次logit回归的标准系数,此处需要小心有可能B_O是自变量前的系数，而B_1是截距项
                B_O_standard = []
                B_1_standard = []
                # 用于存储回归系数的协方差矩阵
                cov11s = []
                cov12s = []
                cov22s = []
                # 用于存储每一次logit回归时系数的P值
                B_0_pValue = []
                B_1_pValue = []
                # 用于存储每一折训练数据的metric与bug的Spearman相关系数
                traingDataSpearman = []
                # 用VARL作为阈值来检测1折testingData数据中，预测的AUC和GM性能如何
                AUCs_VARL = []
                GMs_VARL = []
                # 用LOGIT模型自己原来的度量值代入回归方程得到的预测值，并与训练集Y值进行比较，得到AUC和GM性能，看一下模型的效果如何
                AUCs_predict = []

                # 控制10折次数
                k = 0
                # logit回归的自变量系数为零的次数，若等于10次，则该度量放弃
                B_0_Count = 0
                # 统计pValueLogit[0]大于0.05的次数，若等于10次，则该度量放弃
                pValueCount = 0
                # 分别处理每一折，k控制折数
                for train, test in skf.split(df[metric], df["bugBinary"]):
                    # 9-fold for training Threshold
                    traingData = df.loc[train, [metric, 'bugBinary']]
                    # 1-fold for testing
                    testingData = df.loc[test, [metric, 'bugBinary']]
                    # 需要自行添加逻辑回归所需的intercept变量
                    traingData['intercept'] = 1.0
                    testingData['intercept'] = 1.0

                    # 通过 statsmodels.api 逻辑回归分类; 指定作为训练变量的列，不含目标列`bug`
                    logit = sm.Logit(traingData['bugBinary'], traingData.loc[:, [metric, 'intercept']])
                    # 拟合模型,disp=1 用于显示结果
                    result = logit.fit(method='bfgs', disp=0)

                    pValueLogit = result.pvalues
                    if pValueLogit[0] > 0.05:  # 自变量前的系数
                        pValueCount += 1  # 统计pValueLogit[0]大于0.05的次数，若等于10次，则该度量放弃
                        continue  # 若9-fold 训练数据LOGIT回归系数的P值大于0.05，放弃该数据。

                    B_0_pValue.append(pValueLogit[0])
                    B_1_pValue.append(pValueLogit[1])

                    # 求VARL作为阈值VARL.threshold <- (log(Porbability[1]/Porbability[2])-B[1])/B[2]
                    valueOfbugBinary = traingData["bugBinary"].value_counts()  # 0 和 1 的各自的个数
                    # print("the value of valueOfbugBinary[0] is ", valueOfbugBinary[0])
                    # print("the value of valueOfbugBinary[1] is ", valueOfbugBinary[1])

                    # 用缺陷为大于0的模块数占所有模块之比
                    BaseProbability_1 = valueOfbugBinary[1] / (valueOfbugBinary[0] + valueOfbugBinary[1])
                    B = result.params  # logit回归系数
                    if B[0] == 0:  # 自变量前的系数
                        B_0_Count += 1  # 统计B[0]等于0的次数，若等于10次，则该度量放弃
                        continue  # 若9-fold 训练数据LOGIT回归系数等于0，放弃该数据。
                    B_O.append(B[0])
                    B_1.append(B[1])
                    # 计算标准化系数，此处Y是发生几率比的log值，而它也是断点回归的
                    Y = df.loc[train, metric] * B[0] + B[1]
                    # Y = np.exp(df.loc[train, metric] * B[O] + B[1]) / (1 + np.exp(df.loc[train, metric] * B[O] + B[1]))
                    B_O_standard.append(B[0] / np.std(Y))
                    B_1_standard.append(B[1] / np.std(Y))

                    # 计算VARL阈值及标准差，主要为后续找LATE最大估计值时的阈值做个参照，看是估计值偏大还是偏小，现在并不知道。
                    VARLThreshold = (np.log(BaseProbability_1 / (1 - BaseProbability_1)) - B[1]) / B[0]
                    # 计算LOGIT回归系数矩阵的协方差矩阵，因为计算VARL的标准差要用到,见参考文献[1]
                    cov = result.cov_params()
                    cov11 = cov.iloc[0, 0]
                    cov12 = cov.iloc[0, 1]
                    cov22 = cov.iloc[1, 1]
                    VARLThreshold_se = ((cov.iloc[0, 0] + 2 * VARLThreshold * cov.iloc[0, 1]
                                         + VARLThreshold * VARLThreshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                    VARL_variance = VARLThreshold_se ** 2
                    VARLThresholds.append(VARLThreshold)
                    VARL_variances.append(VARL_variance)

                    cov11s.append(cov11)
                    cov12s.append(cov12)
                    cov22s.append(cov22)

                    # 判断每个度量与bug之间的关系，因为该关系会影响到断点回归时，局部平均处理效应同（LATE）估计值是大于零还是小于零；
                    # 相关系数大于零，则LATE估计值大于零，反之，则LATE估计值小于零;这一步需要在系数和p值判断之后，否则未能删去多余的相关系数
                    traingCorrDf = df.loc[train, [metric, 'bug']].corr('spearman')
                    traingDataSpearman.append(traingCorrDf[metric][1])
                    # 当每个度量与bug之间的相关系数大于零，则正相关，则当前的VARL阈值为最大值，当度量值大于该阈值，则预测为有缺陷；反之，小于。
                    if traingCorrDf[metric][1] > 0:
                        testingData['predictBinary_1'] = testingData[metric].apply(
                            lambda x: 1 if x > VARLThreshold else 0)
                    elif traingCorrDf[metric][1] < 0:
                        testingData['predictBinary_1'] = testingData[metric].apply(
                            lambda x: 1 if x < VARLThreshold else 0)
                    else:
                        continue
                    # 用训练数据集代入预测，检验一下模型的AUC性能如何
                    testingData['predict'] = result.predict(testingData.loc[:, [metric, 'intercept']])
                    # 计算GM性能指标，其公式为GM=(TPR*TNR)**0.5
                    # confusionMatrix = confusion_matrix(testingData["bugBinary"], testingData['predictBinary_1'])
                    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
                    confusionMatrix = confusion_matrix(testingData["bugBinary"], testingData['predictBinary_1'],
                                                       labels=[0, 1])
                    print("confusionMatrix is:\n", confusionMatrix)
                    # 计算混淆矩阵时有些情况只能算出一个TN或TP的
                    # confusionMatrix_predict = confusion_matrix(testingData["bugBinary"], testingData['predict'])
                    tn, fp, fn, tp = confusionMatrix.ravel()
                    # tn_predict, fp_predict, fn_predict, tp_predict = confusionMatrix.ravel()
                    # 当tp且tn为零时，GM值为零，因为GM=(TPR*TNR)**0.5
                    if tp == 0 and tn == 0:
                        GM_VARL = 0
                    elif tp == 0 and tn != 0:
                        GM_VARL = (tn / (tn + fp)) ** 0.5
                    elif tp != 0 and tn == 0:
                        GM_VARL = (tp / (tp + fn)) ** 0.5
                    else:
                        GM_VARL = ((tp / (tp + fn)) * (tn / (tn + fp))) ** 0.5
                    # GM_predict = ((tp_predict/(tp_predict+fn_predict))*(tn_predict/(tn_predict+fp_predict)))**0.5
                    GMs_VARL.append(GM_VARL)

                    # 计算AUC性能指标
                    fpr_VARL, tpr_VARL, thresholds_VARL = roc_curve(testingData["bugBinary"],
                                                                    testingData['predictBinary_1'])
                    fpr_predict, tpr_predict, thresholds_predict = roc_curve(testingData["bugBinary"],
                                                                             testingData['predict'])
                    AUC_VARL = auc(fpr_VARL, tpr_VARL)
                    AUC_predict = auc(fpr_predict, tpr_predict)
                    AUCs_VARL.append(AUC_VARL)
                    AUCs_predict.append(AUC_predict)
                    k += 1
                    # 输出每一折的结果
                    writer.writerow([file, metric, str(k) + "-fold", traingCorrDf[metric][1], 0, B[0], B[1],
                                     pValueLogit[0], pValueLogit[1], cov11, cov12, cov22, VARLThreshold, VARL_variance,
                                     GM_VARL, AUC_VARL, AUC_predict, B[0] / np.std(Y), B[1] / np.std(Y)])

                # 若所有10折训练模型中的自变量参数p值大于0.05，则该度量放弃
                if pValueCount == 10:
                    continue
                # 若所有10折训练模型中的自变量回归系数个数等于0，则该度量也放弃
                if B_0_Count == 10:
                    continue
                # 若某个度量中缺陷数据比较少，导致9折中的缺陷数据为零，这些预测时由于不知道标签信息所以预测性能会失效，故舍去
                if len(GMs_VARL) == 0:
                    continue
                # 分别输出10折中测试集数据预测GM值最大的那一折系数;GM值大于GM平均值那些折的平均值；及所有折的平均值
                # (1)输出10折的平均结果，均值 np.mean()
                writer.writerow([file, metric, "average", np.mean(traingDataSpearman), np.std(traingDataSpearman),
                                 np.mean(B_O), np.mean(B_1), np.mean(B_0_pValue), np.mean(B_1_pValue),
                                 np.mean(cov11s), np.mean(cov12s), np.mean(cov22s), np.mean(VARLThresholds),
                                 np.mean(VARL_variances), np.mean(GMs_VARL), np.mean(AUCs_VARL), np.mean(AUCs_predict)])
                # (2)输出10折中测试集数据预测GM值最大的那一折所有结果;
                print("GMs_VARL is:\n", GMs_VARL)
                maxIndex = GMs_VARL.index(np.max(GMs_VARL))
                writer.writerow([file, metric, "MaxGM-" + str(maxIndex + 1) + "-fold", traingDataSpearman[maxIndex], 0,
                                 B_O[maxIndex], B_1[maxIndex], B_0_pValue[maxIndex], B_1_pValue[maxIndex],
                                 cov11s[maxIndex], cov12s[maxIndex], cov22s[maxIndex], VARLThresholds[maxIndex],
                                 VARL_variances[maxIndex], GMs_VARL[maxIndex], AUCs_VARL[maxIndex],
                                 AUCs_predict[maxIndex]])

                # (3)输出10折中，GM值大于GM平均值那些折的所有结果的平均值；若10折中只有1折符合要求可以输出，则输出与（1）和（2）相同
                traingDataSpearman_GA = []
                greaterAverageIndexList = []  # 此列表用于收集大于GM平均值的列表索引
                B_O_GA = []
                B_1_GA = []
                B_0_pValue_GA = []
                B_1_pValue_GA = []
                cov11s_GA = []
                cov12s_GA = []
                cov22s_GA = []
                VARLThresholds_GA = []
                VARL_variances_GA = []
                GMs_VARL_GA = []
                AUCs_VARL_GA = []
                AUCs_predict_GA = []
                mean_GMs_VARL = np.mean(GMs_VARL)
                for value in GMs_VARL:
                    if value > mean_GMs_VARL:
                        GreaterAverageIndex = GMs_VARL.index(value)
                        greaterAverageIndexList.append(GreaterAverageIndex)
                        traingDataSpearman_GA.append(traingDataSpearman[GreaterAverageIndex])
                        B_O_GA.append(B_O[GreaterAverageIndex])
                        B_1_GA.append(B_1[GreaterAverageIndex])
                        B_0_pValue_GA.append(B_0_pValue[GreaterAverageIndex])
                        B_1_pValue_GA.append(B_1_pValue[GreaterAverageIndex])
                        cov11s_GA.append(cov11s[GreaterAverageIndex])
                        cov12s_GA.append(cov12s[GreaterAverageIndex])
                        cov22s_GA.append(cov22s[GreaterAverageIndex])
                        VARLThresholds_GA.append(VARLThresholds[GreaterAverageIndex])
                        VARL_variances_GA.append(VARL_variances[GreaterAverageIndex])
                        GMs_VARL_GA.append(GMs_VARL[GreaterAverageIndex])
                        AUCs_VARL_GA.append(AUCs_VARL[GreaterAverageIndex])
                        AUCs_predict_GA.append(AUCs_predict[GreaterAverageIndex])

                if len(greaterAverageIndexList) == 0:
                    writer.writerow([file, metric, "greaterAverageGM", traingDataSpearman[maxIndex], 0,
                                     B_O[maxIndex], B_1[maxIndex], B_0_pValue[maxIndex], B_1_pValue[maxIndex],
                                     cov11s[maxIndex], cov12s[maxIndex], cov22s[maxIndex], VARLThresholds[maxIndex],
                                     VARL_variances[maxIndex], GMs_VARL[maxIndex], AUCs_VARL[maxIndex],
                                     AUCs_predict[maxIndex]])
                else:
                    writer.writerow([file, metric, "greaterAverageGM", np.mean(traingDataSpearman_GA),
                                     np.std(traingDataSpearman_GA), np.mean(B_O_GA), np.mean(B_1_GA),
                                     np.mean(B_0_pValue_GA), np.mean(B_1_pValue_GA), np.mean(cov11s_GA),
                                     np.mean(cov12s_GA),
                                     np.mean(cov22s_GA), np.mean(VARLThresholds_GA), np.mean(VARL_variances_GA),
                                     np.mean(GMs_VARL_GA), np.mean(AUCs_VARL_GA), np.mean(AUCs_predict_GA)])

                # 用10折中有效折数的平均值，即np.mean(VARLThresholds)作为阈值，来做断点回归
                threshold = np.mean(VARLThresholds)
                # 用10折中测试集中GM值最大的那一折，即VARLThresholds[maxIndex]作为阈值，来做断点回归
                threshold_maxGM = VARLThresholds[maxIndex]
                # 用10折中GM值大于GM平均值那些折的所有结果的平均值，把这些折的阈值取平均数作为阈值，来做断点回归
                if len(greaterAverageIndexList) == 0:
                    threshold_greaterAverageGM = VARLThresholds[maxIndex]
                    threshold_greaterAverageGM_corr = traingDataSpearman[maxIndex]
                else:
                    threshold_greaterAverageGM = np.mean(VARLThresholds_GA)
                    threshold_greaterAverageGM_corr = np.mean(traingDataSpearman_GA)
                # 输出10折平均的logit回归系数，并用原项目所有数据代入得到预测值
                with open(resultDirectory + metric + "_logitPredict.csv", 'a+', encoding="utf-8", newline='') as f3:
                    writer_f3 = csv.writer(f3)
                    # 用10折中有效折数的平均值的回归系数，来求对应模型所有数据集下预测缺陷概率
                    # 此处B_O和B_1要先除以项目内的Y的标准差来达到统一尺度
                    predictMetirc = np.exp(df.loc[:, [metric]] * np.mean(B_O_standard) + np.mean(B_1_standard)) \
                                    / (1 + np.exp(df.loc[:, [metric]] * np.mean(B_O_standard) + np.mean(B_1_standard)))
                    # 用10折中测试集中GM值最大的那一折的回归系数，来求对应模型所有数据集下预测缺陷概率
                    predictMetirc_maxGM = np.exp(df.loc[:, [metric]] * B_O[maxIndex] + B_1[maxIndex]) \
                                          / (1 + np.exp(df.loc[:, [metric]] * B_O[maxIndex] + B_1[maxIndex]))
                    # 用10折中GM值大于GM平均值那些折的所有结果平均值的回归系数，来求对应模型所有数据集下预测缺陷概率
                    if len(greaterAverageIndexList) == 0:
                        predictMetirc_greaterAverageGM = np.exp(df.loc[:, [metric]] * B_O[maxIndex] + B_1[maxIndex]) \
                                                         / (1 + np.exp(
                            df.loc[:, [metric]] * B_O[maxIndex] + B_1[maxIndex]))
                    else:
                        predictMetirc_greaterAverageGM = np.exp(df.loc[:, [metric]] * np.mean(B_O_GA) + np.mean(B_1_GA)) \
                                                / (1 + np.exp(df.loc[:, [metric]] * np.mean(B_O_GA) + np.mean(B_1_GA)))

                    print("the type of predictMetirc is  ", type(predictMetirc))
                    # 当文件大小为零，即刚创建时，则输出标题
                    if os.path.getsize(resultDirectory + metric + "_logitPredict.csv") == 0:
                        writer_f3.writerow(["fileName", "Name", "bug", "bugBinary", metric, "threshold", "precictValue",
                                            "corr", "threshold_maxGM", "precictValue_maxGM", "corr_threshold_maxGM",
                                            "threshold_greaterAverageGM", "precictValue__greaterAverageGM",
                                            "corr_threshold_greaterAverageGM", "min_Metric", "max_Metric",
                                            "length_Metrix"])
                    # 通过循环输出数据
                    for i in range(len(df)):
                        writer_f3.writerow([file, df.iloc[i, 2], df.loc[i, "bug"], df.loc[i, "bugBinary"],
                                            df.loc[i, metric], threshold, predictMetirc.iloc[i, 0],
                                            np.mean(traingDataSpearman), threshold_maxGM,
                                            predictMetirc_maxGM.iloc[i, 0],
                                            traingDataSpearman[maxIndex], threshold_greaterAverageGM,
                                            predictMetirc_greaterAverageGM.iloc[i, 0], threshold_greaterAverageGM_corr,
                                            np.min(df.loc[:, metric]), np.max(df.loc[:, metric]),
                                            np.max(df.loc[:, metric]) - np.min(df.loc[:, metric])])

    print("this python file name is metrixForRd_tera.py!")


if __name__ == '__main__':
    metrixForRd()
    print("this is mark!")
    pass
