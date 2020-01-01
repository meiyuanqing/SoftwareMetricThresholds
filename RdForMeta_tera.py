'''
v2:把v1版本上的参数整理出来供调用。
bug修复：(1)第764行的pdf["threshold"][k]
        (2)第586行左右的新建四个文件夹的异常处理，（try:）新建前判断其是否存在，若存在则不需要再建，只是里面的图片文件被覆盖

input:(1)读入resultDirectory文件夹中每个度量的断点回归变量X,C和Y,调用R语言包doRdRoubst和LateMaxRdRoubst，求每个度量的阈值；
output:文件夹rdrobustDirectory中以下四个文件,以及四个同名文件夹下断点回归的图片。
      (1)VARLrdrobust.csv
      (2)VARLMaxGMrdrobust.csv
      (3)VARLGreaterAverageGMrdrobust.csv
      (4)MaxLATErdrobust.csv
      以上四个文件主要包括断点回归LATE的估计值及P值，包括以VARL为基准值调整的幅度(最优带宽和移动次数的乘积)。

本代码包为汇总实现断点回归中的第二步，其功能主要有：
    1.对每一个度量文件执行断点回归:有五种阈值作为断点，前三种与VARL相关即分别用VARL的平均值、测试集上GM最大值和超过平均值的VARL值；
      第四种阈值的产生过程：(1)取出metrixForRd.py文件收集到的每一个度量文件里各个度量的取值范围：max(metric)-min(metric)；
                        (2)计算出最优带宽h，从度量值最小值min(metric)开始，依次加最优带宽h长度的端点值作为cutoff；
                        (3)求出(2)中cutoff是否是有效的LATE估计值(p值是否小于0.05)，cutoff存入数组Cs,同时LATE估计值存入数组coef_Cs
                        (4)在(3)的所有有效LATE估计值的数组coef_Cs中，选择最大值时(相同位置)的cutoff作为阈值。
      第四种阈值产生过程的理由是：寻找断点回归的显著的点中，LATE估计量显著的基础上选择最大值为断点，即处理效应最大。

    2.对上述四种阈值进行元分析，最到最终的阈值。

注意事项:
   (1)在安装pip install rpy2==3.2.2模块时，有些头文件可能不在当前venv文件夹下的include文件下,
      这时可去python安装目录下include文件夹里的内容拷贝过来即可。
   (2)文件夹rdrobustDirectory中以下四个文件,以及四个同名文件夹,执行前不能存在，否则报错。

'''

workingDirectory = "/home/mei/RD/terapromise/"
resultDirectory = "/home/mei/RD/terapromise/thresholds/"
rdrobustDirectory = "/home/mei/RD/terapromise/thresholds/rdrobust/"

# 参数说明：
#    (1) rd： 用于存放为断点回归数据（按各个度量名形成文件 ）以及 metricThresholds.csv用入存入LOGIT回归变量的系数协方差，
#             用于计算调整后的阈值计算其方差，为元分析提供数据,默认值是"/home/mei/RD/JURECZKO/thresholds/"。
#    (2) rdd: 用于放断点回归结果的文件的文件夹路径，该路径下包括四个同名子文件夹，用于存入断点回归的图形
def RdForMeta(rd = "/home/mei/RD/terapromise/thresholds/",
              rdd = "/home/mei/RD/terapromise/thresholds/rdrobust/"):

    import os
    import csv
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects

    resultDirectory = rd
    rdrobustDirectory = rdd

    print(os.getcwd())
    os.chdir(resultDirectory)
    print(os.getcwd())

    # creat an R function
    # When it is executed first, the front packages should be installed. For the second time, No need to install anymore
    # Be careful of path in this function: the rdplot output path road is set in the function
    # 主要定义了bandwidth;simpleRdRoubst;doRdRoubst和LateMaxRdRoubst 四个函数
    robjects.r(

        '''
        # install.packages("zoo")
        # install.packages("car")
        # install.packages("carData")
        # install.packages("survival")
        # install.packages("rdrobust")
        # install.packages("stringr")
        
        library("zoo")                # used for rdd package
        library("carData")            # used for rdd package
        library("car")                # used for rdd package
        library("survival")           # used for rdd package    
        library("sandwich")           # used for rdd package
        library("lmtest")             # used for rdd package
        library("AER")                # used for rdd package
        library("Formula")            # used for rdd package
        library("rdd")                # used for IKbandwidth
        library("rdrobust")           # Load RDROBUST package
        library("stringr")            # used for string_c
        library("ROCR")               # compute the auc and ROC curve; used for prediction() and performance()
        
        # 计算最优带宽
        bandwidth <- function(X, C, Y) {    
            #由于python中只有list类型，故用as.vector(unlist(X))函数把list类型转为vector类型
            X <- as.vector(unlist(X))
            C <- as.vector(unlist(C))
            Y <- as.vector(unlist(Y))
            
            y <- Y
            x <- X - C
            result <- list()
            result$rd.bandwidth <- IKbandwidth(x, y)     
            return(result)
        }
        
        # 计算rdrobust中LATE的估计值和p值
        simpleRdRoubst <- function(X, C, Y) {
            #由于python中只有list类型，故用as.vector(unlist(X))函数把list类型转为vector类型
            X <- as.vector(unlist(X))
            C <- as.vector(unlist(C))
            Y <- as.vector(unlist(Y))
            # B <- as.vector(unlist(B))
            y <- Y
            x <- X - C
            bandwidth.RD <- IKbandwidth(x, y)
            result.rdrobust <- rdrobust(y = Y, x = X - C, h = bandwidth.RD, all = TRUE)
            result <- list()
            result$rd.bandwidth <- bandwidth.RD
            
            result$rd.coef <- result.rdrobust$coef
            result$rd.coef.Conventional <- result.rdrobust$coef[1]        #拟输出
            result$rd.coef.BiasCorrected <- result.rdrobust$coef[2]       #拟输出
            result$rd.coef.Robust <- result.rdrobust$coef[3]              #拟输出
        
            result$rd.pv <- result.rdrobust$pv
            result$rd.pv.Conventional <- result.rdrobust$pv[1]            #拟输出
            result$rd.pv.BiasCorrected <- result.rdrobust$pv[2]           #拟输出
            result$rd.pv.Robust <- result.rdrobust$pv[3]                  #拟输出
            
            return(result)
        }    
        
        # 若C断点处LATE不显著，可根据相关系数左移（负相关）或右移（正相关）从而找出C断点最近一处断点，并计算移动距离
        doRdRoubst <- function(X, C, Y, B, N){
            # X <- training.variable[,5]
            # C <- training.variable$threshold
            # Y <- training.variable$precictValue
            # B <- training.variable$bug
            # N <- metricName
        #由于python中只有list类型，故用as.vector(unlist(X))函数把list类型转为vector类型
        X <- as.vector(unlist(X))
        C <- as.vector(unlist(C))
        Y <- as.vector(unlist(Y))
        B <- as.vector(unlist(B))
      
        bugBinary <- 1*(B > 0)  
      
        # 检查X与B之间的spearman相关系数
        corrXB <- cor.test(X, B, method = "spearman")
        corr <- corrXB$estimate
      
        # treatment <- 1*(X > C)
      
        y <- Y
        x <- X - C
        bandwidth.RD <- IKbandwidth(x, y)
    
        # 由于rdroboust包中对带宽的选择用的是CCT方法，对一些度量算不出最优带，所以用rdd包中IKbandwidth函数先算出
        # 按IK算法得出的最优带宽，然后再代入rdrobust包中进行rd回归
      
        # 通过Repeat循环，找到LATE估计值(conventional 和 bias-correct)大于零且其p值小于0.05
        times <- 0
        repeat { 
         
            result.rdrobust <- rdrobust(y = Y, x = X - C + times * bandwidth.RD, h = bandwidth.RD, all = TRUE)
         
            # 若度量与缺陷发生概率正相关，则LATE大于零，当其小于零时，h需要向左移动，即变小，反之，做相反方向处理
            if (corr > 0){
           
            # 若正相关，只要Conventional,Bias-Corrected和Robust中有一个系数大于零，且P值小于0.05
           
            if ((result.rdrobust$coef[1] > 0 & result.rdrobust$pv[1] < 0.05)) {
                break
            }
           
            times <- times + 0.5
           
            }else{
           
            # 若负相关，只要Conventional,Bias-Corrected和Robust中有一个系数大于零，且P值小于0.05
            # |(result.rdrobust$coef[2] < 0 & result.rdrobust$pv[2] < 0.05)|(result.rdrobust$coef[3] < 0 & result.rdrobust$pv[3] < 0.05)
            if ((result.rdrobust$coef[1] < 0 & result.rdrobust$pv[1] < 0.05)) {
                break
            }
           
            times <- times - 0.5
           
            }
        }
      
        # 对DCdensity进行异常判断并处理
        try.DCdensity <- try(DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, bw = bandwidth.RD, plot = FALSE), silent=TRUE)
        if ('try-error' %in% class(try.DCdensity)) {
            McCrary.test <- DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, plot = FALSE)
        } else {
            McCrary.test <- DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, bw = bandwidth.RD, plot = FALSE)
        }
        
        # 输出断点回归的结果，包括该阈值下的各性能指标
        result <- list()
            summary(result.rdrobust)
            # print(str_c(i,"  ","bandwith of the metric is ",bandwidth.RD," !"))
            # print(str_c(i,"  ","result.rdrobust$Estimate of the metric is ",result.rdrobust$Estimate," !"))
            result$times <- times
            result$McCrary.test <- McCrary.test
            result$rd.bandwidth <- bandwidth.RD                           #拟输出
            result$rd.Estimate <- result.rdrobust$Estimate
            result$rd.bws <- result.rdrobust$bws
        
            result$rd.coef <- result.rdrobust$coef
            result$rd.coef.Conventional <- result.rdrobust$coef[1]        #拟输出
            result$rd.coef.BiasCorrected <- result.rdrobust$coef[2]       #拟输出
            result$rd.coef.Robust <- result.rdrobust$coef[3]              #拟输出
        
            result$rd.se <- result.rdrobust$se
            result$rd.se.Conventional <- result.rdrobust$se[1]            #拟输出
            result$rd.se.BiasCorrected <- result.rdrobust$se[2]           #拟输出
            result$rd.se.Robust <- result.rdrobust$se[3]                  #拟输出
        
            result$rd.z <- result.rdrobust$z
        
            result$rd.pv <- result.rdrobust$pv
            result$rd.pv.Conventional <- result.rdrobust$pv[1]            #拟输出
            result$rd.pv.BiasCorrected <- result.rdrobust$pv[2]           #拟输出
            result$rd.pv.Robust <- result.rdrobust$pv[3]                  #拟输出
        
            result$rd.ci <- result.rdrobust$ci                            #拟输出
            result$rd.beta_p_l <- result.rdrobust$beta_p_l
            result$rd.beta_p_r <- result.rdrobust$beta_p_r
            result$rd.V_cl_l <- result.rdrobust$V_cl_l
            result$rd.V_cl_r <- result.rdrobust$V_cl_r
            result$rd.V_rb_l <- result.rdrobust$V_rb_l
            result$rd.V_rb_r <- result.rdrobust$V_rb_r
            result$rd.N <- result.rdrobust$N
            result$rd.Nh <- result.rdrobust$Nh                            #拟输出
            result$rd.Nb <- result.rdrobust$Nb                            #拟输出
            result$rd.tau_cl <- result.rdrobust$tau_cl
            result$rd.tau_bc <- result.rdrobust$tau_bc
            result$rd.c <- result.rdrobust$c
            result$rd.p <- result.rdrobust$p
            result$rd.q <- result.rdrobust$q
            result$rd.bias <- result.rdrobust$bias
            result$rd.kernel <- result.rdrobust$kernel
            result$rd.all <- result.rdrobust$all
            result$rd.vce <- result.rdrobust$vce
            result$rd.bwselect <- result.rdrobust$bwselect              #拟输出
            result$rd.level <- result.rdrobust$level
            result$rd.call <- result.rdrobust$call
        
            # 利用该阈值预测缺陷下的各性能指标
            # treatment <- 1*(X > C - times * bandwidth.RD)
            if (corr > 0){
            
                treatment <- 1*(X > C - times * bandwidth.RD)
            
            } else {
            
                treatment <- 1*(X < C - times * bandwidth.RD)
            
            }
          
            pred <- prediction(treatment,bugBinary)
            auc <- performance(pred,"auc")@y.values
            tpr <- performance(pred,"tpr")@y.values[[1]][2]
            fpr <- performance(pred,"fpr")@y.values[[1]][2]
            tnr <- performance(pred,"tnr")@y.values[[1]][2]
            fnr <- performance(pred,"fnr")@y.values[[1]][2]
            recall <- performance(pred,"rec")@y.values[[1]][2]
            precision <- performance(pred,"prec")@y.values[[1]][2]
            fMeasure <- performance(pred,"f")@y.values[[1]][2]
            GM <- sqrt(tpr*tnr)
        
            result$auc <- auc                       #拟输出
            result$tpr <- tpr
            result$fpr <- fpr
            result$tnr <- tnr
            result$fnr <- fnr
            result$recall <- recall                 #拟输出
            result$precision <- precision           #拟输出
            result$fMeasure <- fMeasure             #拟输出
            result$GM <- GM                         #拟输出
            
             ### restore rdplot to a file named rdplot
             getwd()
             # setwd("/home/mei/RD/JURECZKO/thresholds/rdrobust")
             # getwd()
        
             #  width = 480, height = 480, 这两个参考用于控制图形大小单位是像素
             png(filename = str_c(N,"_rdplot.png"),width = 960, height = 480,)
    
             par(mfrow = c(1,2))
          
             # DCdensity test for X variable
             if ('try-error' %in% class(try.DCdensity)) {
               DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0)
             } else{
               DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, bw = bandwidth.RD)
             }
    
             title(ylab="Density function estimates",xlab="Metric value minus cutoff",main="McCrary.test")
        
             ## rdplot to show rdrobust estimate    未调整过带宽
             # rdplot(y = Y, x = X - C, subset = c(result.rdrobust$h_l, result.rdrobust$h_r),
             #     binselect = "esmv", kernel = "triangular", h = c(result.rdrobust$h_l, result.rdrobust$h_r), p = 1,
             #     title = str_c(N," RD Plot"),
             #     y.label = "the probability of bug",
             #     x.label = "the value of metric")
          
             ## rdplot to show rdrobust estimate   调整过带宽
             rdplot(y = Y, x = X - C + times * bandwidth.RD, subset = c(result.rdrobust$h_l, result.rdrobust$h_r),
                 binselect = "esmv", kernel = "triangular", h = c(result.rdrobust$h_l, result.rdrobust$h_r), p = 1,
                 title = str_c(N," RD Plot with revised h"),
                 y.label = "the probability of bug",
                 x.label = "the value of metric", col.lines = "transparent")
        
             # rect(1, 5, 3, 7, col = "black")
             dev.off()
        
             return(result)
        }
    
    
        # Executing the rdroubst to find the max value of LATE from VARL to the end of metric value
        # (min metric vlue if corr > 0;max metric value if corr < 0)
        # LateMaxRdRoubst(training.variable[,5], training.variable$threshold,
        #            training.variable$precictValue, training.variable$bug){
        # input: (1) X: the metric values             (2) C: the threshold of individual project
        #        (3) Y: the predict value of bug      (4) B: the bug information
        #        (5) N: the name of the metric        (6)minX: training.variable$min_Metric
        #        (7) maxX: training.variable$max_Metric
        LateMaxRdRoubst <- function(X, C, Y, B, N, minX, maxX){
    
        #由于python中只有list类型，故用as.vector(unlist(X))函数把list类型转为vector类型
        X <- as.vector(unlist(X))
        C <- as.vector(unlist(C))
        Y <- as.vector(unlist(Y))
        B <- as.vector(unlist(B))
        minX <- as.vector(unlist(minX))
        maxX <- as.vector(unlist(maxX))
      
        bugBinary <- 1*(B > 0)  
        # 检查X与B之间的spearman相关系数
        corrXB <- cor.test(X,B,method = "spearman")
        corr <- corrXB$estimate
      
        y <- Y
        x <- X - C
        isManual.bandwidth <- 0
        # bandwidth.RD equals 1 when it does not work because of insufficient data
        try.bandwidth.RD <- try(IKbandwidth(x, y), silent=TRUE)
        if ('try-error' %in% class(try.bandwidth.RD)) {
            bandwidth.RD <- 1
            isManual.bandwidth <- 1
        }else{
            bandwidth.RD <- IKbandwidth(x, y)
        }
      
        # 由于rdroboust包中对带宽的选择用的是CCT方法，对一些度量算不出最优带，所以用rdd包中IKbandwidth函数先算出
        # 按IK算法得出的最优带宽，然后再代入rdrobust包中进行rd回归
        # 把所有的LATE估计值放入列表，然后选择列表中最大的LATE估计值来所对应的断点作为阈值。
        # 分两个方向,关于移动幅度的确定：统计每个项目内的信息：cutoff(threshold)-min(X)和max(X)-cutoff(threshold)则左移的距离是
        # cutoff(threshold)-min(X)的最小值，理由是当左移的距离超过该值，则说明cutoff(threshold)-min(X)为最小值的项目，阈值（断点）
        # 左移至度量值最小值之外，这违背了阈值取值范围在度量值的最小值和最大值之间;
        # 同样右移的距离是max(X)-cutoff(threshold)最小值，理由是当右移的距离超过该值，说明max(X)-cutoff(threshold)为最小值的项目，
        # 阈值（断点）右移至度量值最大值之外， 这也违背了阈值取值范围在度量值的最小值和最大值之间。
        # 通过Repeat循环，找到LATE估计值(conventional 和 bias-correct)大于零且其p值小于0.05
        times <- 0
        timesList <- list()
        coefList <- list()
        coefMax <- 0
        result.rdrobust.Max <- list()
    
        repeat { 
        
            try.rdrobust <- try(rdrobust(y = Y, x = X - C + times * bandwidth.RD, h = bandwidth.RD, all = TRUE), silent=TRUE)
            if ('try-error' %in% class(try.rdrobust)) {
                if (corr > 0){
                    if (times > min(C - minX)){
                    break
                    }
                    times <- times + 0.5
                }else{
                    # (- 1) * times > min(maxX - C)
                    if (times < (- 1) * min(maxX - C)){
                      break
                    }
                    times <- times - 0.5
                }
                next
            }else{
                result.rdrobust <- rdrobust(y = Y, x = X - C + times * bandwidth.RD, h = bandwidth.RD, all = TRUE)
            }    
        
            # 若度量与缺陷发生概率正相关，则LATE大于零，当其小于零时，h需要向左移动，即变小，反之，做相反方向处理
            if (corr > 0){
              
                # 若正相关，只要Conventional,Bias-Corrected和Robust中有一个系数大于零，且P值小于0.05      
                if ((result.rdrobust$coef[1] > 0 & result.rdrobust$pv[1] < 0.05)) {
                    timesList <- c(timesList,times)
                    coefList <- c(coefList,result.rdrobust$coef[1])
                    if (result.rdrobust$coef[1] > coefMax){
                        coefMax <- result.rdrobust$coef[1]
                        result.rdrobust.Max <- result.rdrobust
                    }
                    if (times > min(C - minX)){
                      break
                    }    
                }
                times <- times + 0.5      
            }else{
              
                # 若负相关，只要Conventional,Bias-Corrected和Robust中有一个系数大于零，且P值小于0.05
                # |(result.rdrobust$coef[2] < 0 & result.rdrobust$pv[2] < 0.05)|(result.rdrobust$coef[3] < 0 & result.rdrobust$pv[3] < 0.05)
                if ((result.rdrobust$coef[1] < 0 & result.rdrobust$pv[1] < 0.05)) {
                    timesList <- c(timesList,times)
                    coefList <- c(coefList,result.rdrobust$coef[1])
                    if (result.rdrobust$coef[1] < coefMax){
                        coefMax <- result.rdrobust$coef[1]
                        result.rdrobust.Max <- result.rdrobust
                    }
                    if (times < (- 1) * min(maxX - C)){
                        break
                    }
                }
                times <- times - 0.5
            }
        }
    
        # 对DCdensity进行异常判断并处理
        try.DCdensity <- try(DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, bw = bandwidth.RD, plot = FALSE), silent=TRUE)
        if ('try-error' %in% class(try.DCdensity)) {
            try.DCdensity.son <- try(DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, plot = FALSE), silent=TRUE)
            if ('try-error' %in% class(try.DCdensity.son)) {
                 McCrary.test <- 0
            }else{
                McCrary.test <- DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, plot = FALSE)
            }
        } else{
            McCrary.test <- DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, bw = bandwidth.RD, plot = FALSE)
        }
      
        # 输出断点回归的结果，包括该阈值下的各性能指标
        result.rdrobust <- result.rdrobust.Max
        result <- list()
        summary(result.rdrobust)
        result$isManual.bandwidth <- isManual.bandwidth               # the manual bandwidth equals 1
        result$coefMax <- coefMax
        result$timesList <- timesList
        result$times <- times
        result$McCrary.test <- McCrary.test
        result$rd.bandwidth <- bandwidth.RD                           #拟输出
        result$rd.Estimate <- result.rdrobust$Estimate
        result$rd.bws <- result.rdrobust$bws
        
        result$rd.coef <- result.rdrobust$coef
        result$rd.coef.Conventional <- result.rdrobust$coef[1]        #拟输出
        result$rd.coef.BiasCorrected <- result.rdrobust$coef[2]       #拟输出
        result$rd.coef.Robust <- result.rdrobust$coef[3]              #拟输出
        
        result$rd.se <- result.rdrobust$se
        result$rd.se.Conventional <- result.rdrobust$se[1]            #拟输出
        result$rd.se.BiasCorrected <- result.rdrobust$se[2]           #拟输出
        result$rd.se.Robust <- result.rdrobust$se[3]                  #拟输出
        
        result$rd.z <- result.rdrobust$z
        
        result$rd.pv <- result.rdrobust$pv
        result$rd.pv.Conventional <- result.rdrobust$pv[1]            #拟输出
        result$rd.pv.BiasCorrected <- result.rdrobust$pv[2]           #拟输出
        result$rd.pv.Robust <- result.rdrobust$pv[3]                  #拟输出
        
        result$rd.ci <- result.rdrobust$ci                            #拟输出
        result$rd.beta_p_l <- result.rdrobust$beta_p_l
        result$rd.beta_p_r <- result.rdrobust$beta_p_r
        result$rd.V_cl_l <- result.rdrobust$V_cl_l
        result$rd.V_cl_r <- result.rdrobust$V_cl_r
        result$rd.V_rb_l <- result.rdrobust$V_rb_l
        result$rd.V_rb_r <- result.rdrobust$V_rb_r
        result$rd.N <- result.rdrobust$N
        result$rd.Nh <- result.rdrobust$Nh                            #拟输出
        result$rd.Nb <- result.rdrobust$Nb                            #拟输出
        result$rd.tau_cl <- result.rdrobust$tau_cl
        result$rd.tau_bc <- result.rdrobust$tau_bc
        result$rd.c <- result.rdrobust$c
        result$rd.p <- result.rdrobust$p
        result$rd.q <- result.rdrobust$q
        result$rd.bias <- result.rdrobust$bias
        result$rd.kernel <- result.rdrobust$kernel
        result$rd.all <- result.rdrobust$all
        result$rd.vce <- result.rdrobust$vce
        result$rd.bwselect <- result.rdrobust$bwselect              #拟输出
        result$rd.level <- result.rdrobust$level
        result$rd.call <- result.rdrobust$call
      
        # 利用该阈值预测缺陷下的各性能指标
        # treatment <- 1*(X > C - times * bandwidth.RD)
        if (corr > 0){    
            treatment <- 1*(X > C - times * bandwidth.RD)
        }else{
            treatment <- 1*(X < C - times * bandwidth.RD)
        }
        
        pred <- prediction(treatment,bugBinary)
        auc <- performance(pred,"auc")@y.values
        tpr <- performance(pred,"tpr")@y.values[[1]][2]
        fpr <- performance(pred,"fpr")@y.values[[1]][2]
        tnr <- performance(pred,"tnr")@y.values[[1]][2]
        fnr <- performance(pred,"fnr")@y.values[[1]][2]
        recall <- performance(pred,"rec")@y.values[[1]][2]
        precision <- performance(pred,"prec")@y.values[[1]][2]
        fMeasure <- performance(pred,"f")@y.values[[1]][2]
        GM <- sqrt(tpr*tnr)
        
        result$auc <- auc                       #拟输出
        result$tpr <- tpr
        result$fpr <- fpr
        result$tnr <- tnr
        result$fnr <- fnr
        result$recall <- recall                 #拟输出
        result$precision <- precision           #拟输出
        result$fMeasure <- fMeasure             #拟输出
        result$GM <- GM                         #拟输出
        
        ## restore rdplot to a file named rdplot
        getwd()
        # setwd("/home/mei/RD/JURECZKO/thresholds/rdrobust/MaxLATErdrobust")
        # getwd()
        
        #  width = 480, height = 480, 这两个参考用于控制图形大小单位是像素
        png(filename = str_c(N,"_rdplot.png"),width = 480, height = 480,)
        # png(filename = str_c(N,"_rdplot.png"),width = 960, height = 480,)
        # par(pin = c(4.3,4.6))
        # par(mfrow = c(1,2))
        # par(mfrow = c(1,2))
        # par(pin = c(4.3,4.6))
        
        # DCdensity test for X variable
        # if ('try-error' %in% class(try.DCdensity)) {
        #   DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0)
        # } else{
        #   DCdensity(runvar = x - times * bandwidth.RD, cutpoint = 0, bw = bandwidth.RD)
        # }
        #
        # title(ylab="Density function estimates",xlab="Metric value minus cutoff",main="McCrary.test")
        
        # ## rdplot to show rdrobust estimate    未调整过带宽
        # rdplot(y = Y, x = X - C, subset = c(result.rdrobust$h_l, result.rdrobust$h_r),
        #        binselect = "esmv", kernel = "triangular", h = c(result.rdrobust$h_l, result.rdrobust$h_r), p = 1,
        #        title = str_c(N," RD Plot"),
        #        y.label = "the probability of bug",
        #        x.label = "the value of metric")
        
        ## rdplot to show rdrobust estimate   调整过带宽
        # rdplot(y = Y, x = X - C + times * bandwidth.RD, subset = c(result.rdrobust$h_l, result.rdrobust$h_r),
        #        binselect = "esmv", kernel = "triangular", h = c(result.rdrobust$h_l, result.rdrobust$h_r), p = 1,
        #        title = str_c(N," RD Plot with Max LATE Estimate"),
        #        y.label = "the probability of bug",
        #        x.label = "the value of metric")
        
        try.rdplot <- try(rdplot(y = Y, x = X - C + times * bandwidth.RD, subset = c(result.rdrobust$h_l, result.rdrobust$h_r),
                       binselect = "esmv", kernel = "triangular", h = c(result.rdrobust$h_l, result.rdrobust$h_r), p = 1, col.lines = "transparent"), silent=TRUE)
        if (!('try-error' %in% class(try.rdplot))) {
            rdplot(y = Y, x = X - C + times * bandwidth.RD, subset = c(result.rdrobust$h_l, result.rdrobust$h_r),
                   binselect = "esmv", kernel = "triangular", h = c(result.rdrobust$h_l, result.rdrobust$h_r), p = 1,
                   title = str_c(N," RD Plot with Max LATE Estimate"),
                   y.label = "the probability of bug",
                   x.label = "the value of metric", col.lines = "transparent")
        }
        
        # rect(1, 5, 3, 7, col = "black")
        dev.off()
            
        return(result)
        }
    
        f <- function(r) {2 * pi * r}
        
        '''
               )

    # print(robjects.r['f'](3))
    # print(robjects.r['pi'])


    # 把20个度量的“_logitPredict.csv”文件读入predictionFile文件中，分20次，来做每个度量的RD回归
    # (1)os.walk可以用于遍历指定文件下所有的子目录、非目录子文件。
    # (2)os.listdir()用于返回指定的文件夹下包含的文件或文件夹名字的列表，这个列表按字母顺序排序。
    # print(os.listdir(resultDirectory))
    fileList = os.listdir(resultDirectory)
    # print("the prediction list is ", fileList)
    predictionList = []
    for file in fileList:
        if file[-17:] == '_logitPredict.csv':
            predictionList.append(file)

    # 在rd文件夹中创建四个与文件名同名的文件夹用于存入对应断点回归的图
    try:
        os.mkdir(rdrobustDirectory + "VARLrdrobust/")
    except Exception as errVARLrdrobust:
        print(errVARLrdrobust)

    try:
        os.mkdir(rdrobustDirectory + "VARLMaxGMrdrobust/")
    except Exception as errVARLMaxGMrdrobust:
        print(errVARLMaxGMrdrobust)

    try:
        os.mkdir(rdrobustDirectory + "VARLGreaterAverageGMrdrobust/")
    except Exception as errVARLGreaterAverageGMrdrobust:
        print(errVARLGreaterAverageGMrdrobust)

    try:
        os.mkdir(rdrobustDirectory + "MaxLATErdrobust/")
    except Exception as errMaxLATErdrobust:
        print(errMaxLATErdrobust)

    # 在rdd文件夹中创建maxLATEthreshold文件夹，用于存入通过第四种方法找到的阈值，为阈值元分析提供数据
    try:
        os.mkdir(resultDirectory + "maxLATEthreshold/")
    except Exception as errmaxLATEthreshold:
        print(errmaxLATEthreshold)


    # 分20次，做每个度量的RD回归
    i = 0
    for predictionFile in predictionList:
        i += 1
        print("the ", i, " prediction file name is ", predictionFile)
        # 读出每个度量名称
        metricName = predictionFile[-len(predictionFile):-17]

        print("the ", i, " metric name is ", metricName)

        with open(rdrobustDirectory + "VARLrdrobust.csv", 'a+', encoding="utf-8", newline='') as pf1,\
            open(rdrobustDirectory + "VARLMaxGMrdrobust.csv", 'a+', encoding="utf-8", newline='') as pf2,\
            open(rdrobustDirectory + "VARLGreaterAverageGMrdrobust.csv", 'a+', encoding="utf-8", newline='') as pf3,\
            open(rdrobustDirectory + "MaxLATErdrobust.csv", 'a+', encoding="utf-8", newline='') as pf4:

            writer_pf1 = csv.writer(pf1)
            writer_pf2 = csv.writer(pf2)
            writer_pf3 = csv.writer(pf3)
            writer_pf4 = csv.writer(pf4)
            # 输出VARLrdrobust.csv"的标题行
            if os.path.getsize(rdrobustDirectory + "VARLrdrobust.csv") == 0:
                writer_pf1.writerow(["metricName", "McCrary_Test", "coef_Conventional", "pv_Conventional", "coef_BiasCorrected",
                                 "pv_BiasCorrected", "coef_Robust", "pv_Robust", "coef_Conventional_ci_low",
                                 "coef_Conventional_ci_up", "rd_bwselect", "rd_bandwidth", "times", "Recall",
                                 "Precision", "fMeasure", "AUC", "GM", "VARL_Threshold","VARL_Threshold_revised",
                                 "se.Conventional", "se.BiasCorrected", "se.Robust"])
            # 输出VARLMaxGMrdrobust.csv"的标题行
            if os.path.getsize(rdrobustDirectory + "VARLMaxGMrdrobust.csv") == 0:
                writer_pf2.writerow(["metricName", "McCrary_Test", "coef_Conventional", "pv_Conventional", "coef_BiasCorrected",
                                 "pv_BiasCorrected", "coef_Robust", "pv_Robust", "coef_Conventional_ci_low",
                                 "coef_Conventional_ci_up", "rd_bwselect", "rd_bandwidth", "times", "Recall",
                                 "Precision", "fMeasure", "AUC", "GM", "VARL_Threshold","VARL_Threshold_revised",
                                 "se.Conventional", "se.BiasCorrected", "se.Robust"])
            # 输出VARLGreaterAverageGMrdrobust.csv"的标题行
            if os.path.getsize(rdrobustDirectory + "VARLGreaterAverageGMrdrobust.csv") == 0:
                writer_pf3.writerow(["metricName", "McCrary_Test", "coef_Conventional", "pv_Conventional", "coef_BiasCorrected",
                                 "pv_BiasCorrected", "coef_Robust", "pv_Robust", "coef_Conventional_ci_low",
                                 "coef_Conventional_ci_up", "rd_bwselect", "rd_bandwidth", "times", "Recall",
                                 "Precision", "fMeasure", "AUC", "GM", "VARL_Threshold","VARL_Threshold_revised",
                                 "se.Conventional", "se.BiasCorrected", "se.Robust"])

            # 输出MaxLATErdrobust.csv"的标题行
            if os.path.getsize(rdrobustDirectory + "MaxLATErdrobust.csv") == 0:
                writer_pf4.writerow(["metricName", "McCrary_Test", "coef_Conventional", "pv_Conventional", "coef_BiasCorrected",
                                 "pv_BiasCorrected", "coef_Robust", "pv_Robust", "coef_Conventional_ci_low",
                                 "coef_Conventional_ci_up", "rd_bwselect", "rd_bandwidth", "times", "Recall", "Precision",
                                 "fMeasure", "AUC", "GM", "isManual.bandwidth", "VARL_Threshold",
                                 "VARL_Threshold_revised", "se.Conventional", "se.BiasCorrected", "se.Robust"])

            print(os.getcwd())
            # 读入一个项目
            pdf = pd.read_csv(resultDirectory + predictionFile)

            # 把dataframe格式转成list格式： np.array(X).tolist()，断点回归函数的参数格式是list格式，故转换，下同
            X = np.array(pdf[metricName]).tolist()
            C = np.array(pdf["threshold"]).tolist()                          # 10折中有效折数的VARL平均值作阈值
            Y = np.array(pdf["precictValue"]).tolist()
            B = np.array(pdf["bug"]).tolist()
            N = metricName
            minX = np.array(pdf["min_Metric"]).tolist()
            maxX = np.array(pdf["max_Metric"]).tolist()

            C_VARLMaxGM = np.array(pdf["threshold_maxGM"]).tolist()                # 10折中测试集上GM性能最大的那一折的VARL作为阈值
            C_VARLGreaterAverageGM = np.array(pdf["threshold_greaterAverageGM"]).tolist()     # 10折中测试集上GM性能超过平均GM的VARL作为阈值

            # 输出VARLMaxGMrdrobust.csv"文件中各度量回归结果
            # 由于有些度量可能没有充足数据计算带宽（Insufficient data in the calculated bandwidth.），进而影响断点回归，通过异常机制解决
            # Rd的结果输出有两种：(1)根据Rd.names中名称的顺序i用Rd[i]输出，需要小心每个顺序，还要查看名称列表，不方便阅读；
            #                  (2)通过Rd.rx(name)输出，例如：Rd.rx('times')输出向量名为‘times’的值；
            #                     若是该值是矩阵，则用Rd.rx2('a').rx(1, 1) # first element of 'a'；
            #                     tmp.rx2('a').rx(1, True)  # first row of 'a'
            try:
                print(os.getcwd())
                os.chdir(rdrobustDirectory + "VARLrdrobust/")       # 调整路径，用于输出doRdRoubst函数中的图像文件。
                print(os.getcwd())
                Rd = robjects.r['doRdRoubst'](X, C, Y, B, N)

                # 输出R函数doRdRoubst的结果
                writer_pf1.writerow([metricName, robjects.r['as.numeric'](Rd.rx("McCrary.test"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 1))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 2))[0],
                                     robjects.r['as.character'](Rd.rx("rd.bwselect"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.bandwidth"))[0],
                                     robjects.r['as.numeric'](Rd.rx("times"))[0],
                                     robjects.r['as.numeric'](Rd.rx("recall"))[0],
                                     robjects.r['as.numeric'](Rd.rx("precision"))[0],
                                     robjects.r['as.numeric'](Rd.rx("fMeasure"))[0],
                                     robjects.r['as.numeric'](Rd.rx("auc")[0][0])[0],         #由于auc是一个二维list
                                     robjects.r['as.numeric'](Rd.rx("GM"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Robust"))[0]])
            except Exception as err1:
                print(err1)

            # 输出VARLMaxGMrdrobust.csv"文件中各度量回归结果
            try:
                print(os.getcwd())
                os.chdir(rdrobustDirectory + "VARLMaxGMrdrobust/")         # 调整路径，用于输出doRdRoubst函数中的图像文件。
                print(os.getcwd())
                Rd = robjects.r['doRdRoubst'](X, C_VARLMaxGM, Y, B, N)

                # 输出R函数doRdRoubst的结果
                writer_pf2.writerow([metricName, robjects.r['as.numeric'](Rd.rx("McCrary.test"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 1))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 2))[0],
                                     robjects.r['as.character'](Rd.rx("rd.bwselect"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.bandwidth"))[0],
                                     robjects.r['as.numeric'](Rd.rx("times"))[0],
                                     robjects.r['as.numeric'](Rd.rx("recall"))[0],
                                     robjects.r['as.numeric'](Rd.rx("precision"))[0],
                                     robjects.r['as.numeric'](Rd.rx("fMeasure"))[0],
                                     robjects.r['as.numeric'](Rd.rx("auc")[0][0])[0],         #由于auc是一个二维list
                                     robjects.r['as.numeric'](Rd.rx("GM"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Robust"))[0]])
            except Exception as err2:
                print(err2)

            # 输出VARLGreaterAverageGMrdrobust.csv"文件中各度量回归结果
            try:
                print(os.getcwd())
                os.chdir(rdrobustDirectory + "VARLGreaterAverageGMrdrobust/")     # 调整路径，用于输出doRdRoubst函数中的图像文件。
                print(os.getcwd())
                Rd = robjects.r['doRdRoubst'](X, C_VARLGreaterAverageGM, Y, B, N)

                # 输出R函数doRdRoubst的结果
                writer_pf3.writerow([metricName, robjects.r['as.numeric'](Rd.rx("McCrary.test"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 1))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 2))[0],
                                     robjects.r['as.character'](Rd.rx("rd.bwselect"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.bandwidth"))[0],
                                     robjects.r['as.numeric'](Rd.rx("times"))[0],
                                     robjects.r['as.numeric'](Rd.rx("recall"))[0],
                                     robjects.r['as.numeric'](Rd.rx("precision"))[0],
                                     robjects.r['as.numeric'](Rd.rx("fMeasure"))[0],
                                     robjects.r['as.numeric'](Rd.rx("auc")[0][0])[0],         #由于auc是一个二维list
                                     robjects.r['as.numeric'](Rd.rx("GM"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Robust"))[0]])
            except Exception as err3:
                print(err3)

            # 第四种阈值从VARL值开始，若相关系数大于零，每次减小h，直到cutoff到度量最小值，反之，每次增加h,直到cutoff到度量最大值
            # 输出MaxLATErdrobust.csv"文件中各度量回归结果
            try:
                print(os.getcwd())
                os.chdir(rdrobustDirectory + "MaxLATErdrobust/")     # 调整路径，用于输出doRdRoubst函数中的图像文件。
                print(os.getcwd())
                Rd = robjects.r['LateMaxRdRoubst'](X, C, Y, B, N, minX, maxX)

                # threshold_MaxLATE这一列输出到每个度量的_logitPredict.csv文件(新)中，与前三个阈值一起做元分析。
                # os.path.dirname(__file__)表示当前文件绝对路径；os.path.join()把绝对路径和文件名合并
                k = 0         # 控制threshold_MaxLATE行数(k)输出
                with open(resultDirectory + predictionFile, 'r', encoding="ISO-8859-1") as csvFile:
                    rows = csv.reader(csvFile)
                    fieldnamesCsvFile = next(rows)            # 获取数据的第一行标题栏，next方法获取
                    fieldnamesCsvFile.append("threshold_MaxLATE")
                    with open(resultDirectory + "maxLATEthreshold/" + "new_" + predictionFile, 'w') as newf:
                        writer = csv.writer(newf)
                        writer.writerow(fieldnamesCsvFile)    # 写入标题栏
                        for row in rows:         # 写入除标题栏剩下的数据
                            row.append(pdf["threshold"][k] - robjects.r['as.numeric'](Rd.rx("times"))[0]
                                       * robjects.r['as.numeric'](Rd.rx("rd.bandwidth"))[0])
                            writer.writerow(row)
                            k += 1

                # 输出R函数doRdRoubst的结果
                writer_pf4.writerow([metricName, robjects.r['as.numeric'](Rd.rx("McCrary.test"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.coef.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.pv.Robust"))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 1))[0],
                                     robjects.r['as.numeric'](Rd.rx2("rd.ci").rx(1, 2))[0],
                                     robjects.r['as.character'](Rd.rx("rd.bwselect"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.bandwidth"))[0],
                                     robjects.r['as.numeric'](Rd.rx("times"))[0],
                                     robjects.r['as.numeric'](Rd.rx("recall"))[0],
                                     robjects.r['as.numeric'](Rd.rx("precision"))[0],
                                     robjects.r['as.numeric'](Rd.rx("fMeasure"))[0],
                                     robjects.r['as.numeric'](Rd.rx("auc")[0][0])[0],         #由于auc是一个二维list
                                     robjects.r['as.numeric'](Rd.rx("GM"))[0],
                                     robjects.r['as.numeric'](Rd.rx("isManual.bandwidth"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Conventional"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.BiasCorrected"))[0],
                                     robjects.r['as.numeric'](Rd.rx("rd.se.Robust"))[0]])

            except Exception as err7:
                print(err7)

        # if i == 1:
        #     break

if __name__ == '__main__':
    RdForMeta()
    pass