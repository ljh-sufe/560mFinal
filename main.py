
'''
https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/code
'''

import pandas as pd
import numpy as np
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import count,when,isnull,isnan,col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Normalizer, StandardScaler
from pyspark.sql.types import StructType,StructField,StringType
spark  = SparkSession.builder.master("local[*]").appName("final").getOrCreate()
sc = spark.sparkContext

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #
    #
    # trainDf = spark.read.option('header','true').csv('data/train_timeseries/train_timeseries.csv',
    #     inferSchema=True)
    # trainDf.printSchema()
    # trainDf.describe().show()
    # trainDf.show(5,truncate=3)
    #
    #
    # ### for the target 'score', fill na with 0
    # trainDf = trainDf.fillna({"score":0.0})
    # # df=trainDf
    # # df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show() # no nan value
    #
    # ### fips int to string
    # trainDf = trainDf.withColumn("fips_str", col("fips").cast("string")).drop("fips")
    #
    # ### mean variance scaler
    #
    # meanScore = trainDf.agg({"score":"mean"}).collect()[0][0]
    # stdScore = trainDf.agg({"score":"stddev"}).collect()[0][0]
    #
    # trainDf = trainDf.withColumn("norm_score",(trainDf["score"] - meanScore) / stdScore)
    #
    #
    # from pyspark.ml.regression import LinearRegression
    # from pyspark.ml.feature import StringIndexer, VectorAssembler
    #
    # feature = VectorAssembler(inputCols=["PRECTOT", "PS", "QV2M", "T2M", "T2MDEW","T2MWET",
    #                                      "T2M_MAX","T2M_MIN","T2M_RANGE","TS","WS10M","WS10M_MAX",
    #                                      "WS10M_MIN","WS10M_RANGE","WS50M","WS50M_MAX","WS50M_MIN","WS50M_RANGE"],
    #                           outputCol="features")
    # feature_vector = feature.transform(trainDf)
    # lr = LinearRegression(featuresCol="features",
    #                       labelCol="norm_score",
    #                       elasticNetParam=0,  ## 0 L2(ridge)  and 1 as L1(lasso)
    #                       regParam=1  ##regularization para
    #                     )
    #
    # lrModel = lr.fit(feature_vector)
    #
    # trainingSummary = lrModel.summary
    # print("numIterations: %d" % trainingSummary.totalIterations)
    # print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    # # trainingSummary.residuals.show()
    # print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    # print("r2: %f" % trainingSummary.r2)
    # trainingSummary.meanAbsoluteError
    # lrModel.coefficients
    # lrModel.intercept
    #
    #
    # trainDf_pd = trainDf.toPandas()
    trainDf_pd = pd.read_pickle("data/train_timeseries/train_timeseries.pkl")
    trainDf_pd = trainDf_pd[trainDf_pd["fips"]==1001]
    # trainDf_pd = trainDf_pd.dropna()

    meanScore=trainDf_pd.groupby("fips").apply(lambda x: x["score"].mean()).to_dict()
    stdScore=trainDf_pd.groupby("fips").apply(lambda x: x["score"].std()).to_dict()
    f = lambda x,y: (x-meanScore[y])/stdScore[y]
    a=trainDf_pd.groupby("fips").apply(lambda x: f(x["score"], x["fips"].iloc[0]))
    trainDf_pd["demean_score"] = a.values

    from sklearn.linear_model import LinearRegression,Ridge, Lasso

    X = trainDf_pd[["PRECTOT", "PS", "QV2M", "T2M", "T2MDEW","T2MWET",
                    "T2M_MAX","T2M_MIN","T2M_RANGE","TS","WS10M","WS10M_MAX",
                    "WS10M_MIN","WS10M_RANGE","WS50M","WS50M_MAX","WS50M_MIN","WS50M_RANGE"]]

    dflist = [trainDf_pd]
    features=X.columns.to_list()
    for i in [1,2,3,4,5,6]:
        lasti = X.shift(i)
        newFeatures = [x+"_Dm"+str(i) for x in X.columns]
        lasti.columns= newFeatures
        dflist.append(lasti)
        features.extend(newFeatures)

    finalDf=pd.concat(dflist, axis=1)
    finalDf=finalDf.dropna(subset="score")
    finalDf=finalDf.dropna()


    import datetime

    finalDf["date"].map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    finalDf["month"]=pd.Series(pd.DatetimeIndex(finalDf["date"])).map(lambda x: x.month).values
    monthMeanScore = finalDf.groupby("month").apply(lambda x: x["score"].mean()).to_dict()
    monthStdScore = finalDf.groupby("month").apply(lambda x: x["score"].std()).to_dict()
    f = lambda x, y: (x - monthMeanScore[y]) / monthStdScore[y]
    a = finalDf.groupby("month").apply(lambda x: f(x["score"], x["month"].iloc[0]))
    finalDf["demean_score"] = a.values

    X = finalDf[features]
    y = finalDf["score"]

    X = (X-X.mean())/X.std()
    # y=(y-y.mean())/y.std()
    a=finalDf.corr()

    finalDf[["T2M","T2M_Dm1","T2M_Dm2","T2M_Dm3","T2M_Dm4","T2M_Dm5","T2M_Dm6"]].corr()


    reg = Ridge().fit(X, y)
    reg.score(X, y)

    y_pred = reg.predict(X)
    (y-y_pred).abs().mean()

    reg.coef_