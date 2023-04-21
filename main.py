

import pandas as pd
import numpy as np
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import count,when,isnull,isnan,col
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.types import StructType,StructField,StringType
spark  = SparkSession.builder.master("local[*]").appName("final").getOrCreate()
sc = spark.sparkContext

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    trainDf = spark.read.option('header','true').csv('data/train_timeseries/train_timeseries.csv',
        inferSchema=True)
    trainDf.printSchema()
    trainDf.describe().show()
    trainDf.show(5,truncate=3)


    ### for the target 'score', fill na with 0
    trainDf = trainDf.fillna({"score":0.0})
    # df=trainDf
    # df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show() # no nan value

    ### fips int to string
    trainDf = trainDf.withColumn("fips_str", col("fips").cast("string")).drop("fips")


    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.feature import StringIndexer, VectorAssembler

    feature = VectorAssembler(inputCols=["PRECTOT", "PS", "QV2M", "T2M", "T2MDEW","T2MWET",
                                         "T2M_MAX","T2M_MIN","T2M_RANGE","TS","WS10M","WS10M_MAX",
                                         "WS10M_MIN","WS10M_RANGE","WS50M","WS50M_MAX","WS50M_MIN","WS50M_RANGE"],
                              outputCol="features")
    feature_vector = feature.transform(trainDf)
    lr = LinearRegression(featuresCol="features",
                          labelCol="score")

    lrModel = lr.fit(feature_vector)

    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    # trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    lrModel.coefficients
    lrModel.intercept