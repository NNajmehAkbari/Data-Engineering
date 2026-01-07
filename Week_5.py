# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 Data-Intensive Programming, Exercise 5
# MAGIC
# MAGIC This exercise is in three parts.
# MAGIC
# MAGIC - Tasks 1-4 are tasks related to using the Spark machine learning (ML) library.
# MAGIC - Tasks 5-7 contain tasks for aggregated data frames with streaming data.
# MAGIC - Task 8 is a theory question related to the lectures.
# MAGIC
# MAGIC This is the **Python** version, switch to the Scala version if you want to do the tasks in Scala.
# MAGIC
# MAGIC Each task has its own cell(s) for the code. Add your solutions to the cells. You are free to add more cells if you feel it is necessary.<br>
# MAGIC There are cells with test code and example output following most of the tasks that involve producing code.
# MAGIC
# MAGIC At the end of the notebook, there is a question regarding the use of AI or other collaboration when working the tasks.<br>
# MAGIC Please remember to answer the AI question. And finally, don't forget to submit your solutions to Moodle: [Weekly Exercise #5](https://moodle.tuni.fi/mod/assign/view.php?id=3503820)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Some resources that can help with the tasks in this exercise:
# MAGIC
# MAGIC - The [tutorial notebook](https://adb-7895492183558578.18.azuredatabricks.net/editor/notebooks/743402606902162) from our course
# MAGIC - Chapters 8 and 10 in [Learning Spark, 2nd Edition](https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
# MAGIC     - There are additional code examples in the related [GitHub repository](https://github.com/databricks/LearningSparkV2).
# MAGIC     - The book related notebooks can be imported to Databricks by choosing `import` in your workspace and using the URL<br> `https://github.com/databricks/LearningSparkV2/blob/master/notebooks/LearningSparkv2.dbc`
# MAGIC - [Apache Spark documentation](https://spark.apache.org/docs/3.5.6/sql-ref-functions.html) on all available functions that can be used on DataFrames.<br>
# MAGIC   The full [Spark Scala functions API listing](https://spark.apache.org/docs/3.5.6/api/scala/org/apache/spark/sql/functions$.html) for the functions package might have some additional functions listed that have not been updated in the documentation.

# COMMAND ----------

# some imports that might be required in the tasks
import time
from typing import List

from pyspark.rdd import RDD

from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Row
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.types import DoubleType, StructField, StructType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# COMMAND ----------

# helper function for removing all files from a given folder
def removeFiles(folder: str) -> None:
    try:
        files = dbutils.fs.ls(folder)
        for file in files:
            dbutils.fs.rm(file.path)
    except Exception:
        # the folder did not exist => do nothing
        pass

# COMMAND ----------

# the initial data for the linear regression tasks
hugeSequenceOfXYData: list[Row] = [
    (9.44, 14.41), (0.89, 1.77), (8.65, 12.47), (10.43, 15.43), (7.39, 11.03), (10.06, 15.18), (2.07, 3.19), (1.24, 1.45),
    (3.84, 5.45), (10.78, 16.51), (10.23, 16.11), (9.32, 13.96), (7.98, 12.32), (0.99, 1.02), (6.85, 9.62), (8.59, 13.39),
    (7.35, 10.44), (9.85, 15.26), (4.59, 7.26), (2.43, 3.35), (1.58, 2.71), (1.59, 2.2), (2.1, 2.95), (0.62, 0.47),
    (5.65, 9.02), (5.9, 9.58), (8.5, 12.39), (8.74, 13.73), (1.93, 3.37), (10.22, 15.03), (10.25, 15.63), (1.97, 2.96),
    (8.03, 12.03), (2.05, 3.23), (0.69, 0.9), (7.58, 11.01), (9.99, 14.83), (10.53, 15.92), (6.12, 9.48), (1.34, 2.83),
    (3.87, 5.27), (4.98, 7.21), (4.72, 6.48), (8.15, 12.19), (2.37, 3.45), (10.19, 15.16), (10.28, 15.39), (8.6, 12.76),
    (7.46, 11.11), (0.25, 0.41), (6.41, 9.55), (10.49, 15.61), (5.18, 7.92), (3.74, 6.18), (6.27, 9.25), (7.51, 11.11),
    (4.07, 6.63), (5.17, 6.95), (9.61, 14.85), (4.17, 6.31), (4.12, 6.31), (9.22, 13.96), (5.54, 8.2), (0.58, 0.46),
    (10.13, 14.68), (0.53, 1.25), (6.87, 10.0), (7.17, 10.35), (0.09, -0.55), (10.8, 16.6), (10.31, 15.96), (4.74, 6.53),
    (1.6, 2.31), (5.45, 7.84), (0.65, 1.02), (2.89, 3.93), (6.28, 9.21), (8.59, 13.05), (6.6, 10.51), (8.42, 12.91)
]
dataRows: list[Row] = [Row(x, y) for x, y in hugeSequenceOfXYData]
dataRDD: RDD[Row] = spark.sparkContext.parallelize(dataRows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Linear regression - Training and test data
# MAGIC
# MAGIC ### Background
# MAGIC
# MAGIC In statistics, Simple linear regression is a linear regression model with a single explanatory variable.
# MAGIC That is, it concerns two-dimensional sample points with one independent variable and one dependent variable
# MAGIC (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line)
# MAGIC that, as accurately as possible, predicts the dependent variable values as a function of the independent variable.
# MAGIC The adjective simple refers to the fact that the outcome variable is related to a single predictor. Wikipedia: [Simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
# MAGIC
# MAGIC You are given an RDD of Rows, `dataRDD`, where the first element are the `x` and the second the `y` values.<br>
# MAGIC We are aiming at finding a simple linear regression model for the dataset using Spark ML library. I.e. find a function `f` so that `y = f(x)` (for the 2-dimensional case `f(x)=ax+b`).
# MAGIC
# MAGIC ### Task instructions
# MAGIC
# MAGIC Transform the given `dataRDD` to a DataFrame `dataDF`, with two columns `X` (of type Double) and `label` (of type Double).
# MAGIC (`label` used here because that is the default dependent variable name in Spark ML library)
# MAGIC
# MAGIC Then split the rows in the data frame into training and testing data frames.

# COMMAND ----------

# StructType like a template for the DataFrame.
# False → The column cannot be null

schema = StructType([
    StructField("X", DoubleType(), False),
    StructField("label", DoubleType(), False)
])

# COMMAND ----------

dataDF: DataFrame = spark.createDataFrame(dataRDD, schema)
trainingDF, testDF = dataDF.randomSplit([0.80, 0.2], seed=1)


# COMMAND ----------

# Standardization is mostly useful when:
# We have many features
# Those features are on very different scales
# The solver uses gradient descent (like L-BFGS)

dataDF.select('X').describe().show()

# COMMAND ----------

print(f"Training set size: {trainingDF.count()}")
print(f"Test set size: {testDF.count()}")
print("Training set (showing only the first 6 points):")
trainingDF.limit(6).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC Training set size: 67
# MAGIC Test set size: 13
# MAGIC Training set (showing only the first 6 points):
# MAGIC +----+-----+
# MAGIC |   X|label|
# MAGIC +----+-----+
# MAGIC |0.89| 1.77|
# MAGIC |1.24| 1.45|
# MAGIC |2.07| 3.19|
# MAGIC |3.84| 5.45|
# MAGIC |8.65|12.47|
# MAGIC |9.44|14.41|
# MAGIC +----+-----+
# MAGIC ```
# MAGIC
# MAGIC The data splitting for the example output was done by using a seed value of 1 for the data frames random splitting method.<bR>
# MAGIC Your output does not have to match this exactly, not even with the sizes of the training and test sets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Linear regression - Training the model
# MAGIC
# MAGIC To be able to use the ML algorithms in Spark, the input data must be given as a vector in one column. To make it easy to transform the input data into this vector format, Spark offers VectorAssembler objects.
# MAGIC
# MAGIC - Create a `VectorAssembler` for mapping the input column `X` to `features` column. And apply it to training data frame, `trainingDF,` in order to create an assembled training data frame.
# MAGIC - Then create a `LinearRegression` object. And use it with the assembled training data frame to train a linear regression model.

# COMMAND ----------

# VectorAssembler is a Spark ML tool that combines one or more columns 
# into a single vector column called "features".
# Machine learning models in Spark always expect the data in vector form.
# A	B C	D label	---> features
# 1	5 7	2 0	         [1, 5, 7, 2] All in one vector

vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["X"],
                                                   outputCol="features")

assembledTrainingDF: DataFrame = vectorAssembler.transform(trainingDF)

# COMMAND ----------

# print the schema and the first 6 rows of the assembled data frame:
assembledTrainingDF.printSchema()
assembledTrainingDF.limit(6).show()

# COMMAND ----------

lr: LinearRegression = LinearRegression(
    featuresCol="features",
    labelCol="label",
    standardization=False,
    solver="normal"
)


lrModel: LinearRegressionModel = lr.fit(assembledTrainingDF)

# COMMAND ----------

# print out a sample of the predictions
lrModel.summary.predictions.limit(6).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC root
# MAGIC  |-- X: double (nullable = false)
# MAGIC  |-- label: double (nullable = false)
# MAGIC  |-- features: vector (nullable = true)
# MAGIC
# MAGIC +----+-----+--------+
# MAGIC |   X|label|features|
# MAGIC +----+-----+--------+
# MAGIC |0.89| 1.77|  [0.89]|
# MAGIC |1.24| 1.45|  [1.24]|
# MAGIC |2.07| 3.19|  [2.07]|
# MAGIC |3.84| 5.45|  [3.84]|
# MAGIC |8.65|12.47|  [8.65]|
# MAGIC |9.44|14.41|  [9.44]|
# MAGIC +----+-----+--------+
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```text
# MAGIC +----+-----+--------+------------------+
# MAGIC |   X|label|features|        prediction|
# MAGIC +----+-----+--------+------------------+
# MAGIC |0.89| 1.77|  [0.89]|1.2631112943485854|
# MAGIC |1.24| 1.45|  [1.24]| 1.794035541443395|
# MAGIC |2.07| 3.19|  [2.07]| 3.053084470268229|
# MAGIC |3.84| 5.45|  [3.84]|5.7380442341476945|
# MAGIC |8.65|12.47|  [8.65]| 13.03446031565065|
# MAGIC |9.44|14.41|  [9.44]|14.232832187664647|
# MAGIC +----+-----+--------+------------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Linear regression - Test the model
# MAGIC
# MAGIC Apply the trained linear regression model from task 2 to the test dataset.
# MAGIC
# MAGIC Then calculate the RMSE (root mean square error) for the test dataset predictions using `RegressionEvaluator` from Spark ML library.
# MAGIC
# MAGIC (The Python cell after the example output can be used to visualize the linear regression tasks)

# COMMAND ----------

# I tried different spellings and different seeds, 
# but I wasn’t able to get a better result than the current one.
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

assembledTestingDF = vectorAssembler.transform(testDF)
testPredictions: DataFrame = lrModel.transform(assembledTestingDF)
testError: float = evaluator.evaluate(testPredictions)


# COMMAND ----------

print(f"The RMSE for the model is {testError}")
testPredictions.limit(6).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The RMSE for the model is 0.3873862940376186
# MAGIC +----+-----+--------+------------------+
# MAGIC |   X|label|features|        prediction|
# MAGIC +----+-----+--------+------------------+
# MAGIC |7.39|11.03|  [7.39]|11.123133026109334|
# MAGIC |7.35|10.44|  [7.35]|  11.0624559692985|
# MAGIC |1.59|  2.2|  [1.59]|2.3249597885382047|
# MAGIC | 5.9| 9.58|   [5.9]| 8.862912659905717|
# MAGIC |0.69|  0.9|  [0.69]|0.9597260102944085|
# MAGIC |9.99|14.83|  [9.99]|15.067141718813636|
# MAGIC +----+-----+--------+------------------+
# MAGIC ```
# MAGIC
# MAGIC
# MAGIC 0 for RMSE would indicate perfect fit and the more deviations there are the larger the RMSE will be.<br>
# MAGIC You can try a different seed for dividing the data and different parameters for the linear regression to get different results.

# COMMAND ----------

# Full visualization of the linear regression exercise (can be done since there are only limited number of source data points)
# Using matplotlib library to plot the data points and the prediction line
from matplotlib import pyplot

training_data = trainingDF.select("X", "label").collect()
test_data = testDF.select("X", "label").collect()
test_predictions = testPredictions.select("X", "prediction").collect()

pyplot.xlabel("x")
pyplot.ylabel("y")
pyplot.xlim(-1.0, 12.0)
pyplot.ylim(-1.0, 17.0)
pyplot.plot([row[0] for row in training_data], [row[1] for row in training_data], 'bo')
pyplot.plot([row[0] for row in test_data], [row[1] for row in test_data], 'go')
pyplot.plot([row[0] for row in test_predictions], [row[1] for row in test_predictions], 'r-')
pyplot.legend(["training data", "test data", "prediction line"])
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Linear regression - Making new predictions
# MAGIC
# MAGIC Use the trained `LinearRegressionModel` from task 2 to predict the `y` values for the following `x` values:<br>
# MAGIC `-2.8`, `3.14`, `9.9`, `11.11`, `22.22`, `123.45`.

# COMMAND ----------

# Spark’s createDataFrame() expects each row to be a tuple (or list).
# To make a tuple with one element, Python requires a comma.

df = spark.createDataFrame(
    [(-2.8,), (3.14,), (9.9,), (11.11,), (22.22,), (123.45,)],
    ["X"]
)
newAssembledDF = vectorAssembler.transform(df)
newPredictions = lrModel.transform(newAssembledDF)


# COMMAND ----------

newPredictions.select("X", "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC +------+------------------+
# MAGIC |     X|        prediction|
# MAGIC +------+------------------+
# MAGIC |  -2.8|-4.334347196450978|
# MAGIC |  3.14| 4.676195739958076|
# MAGIC |   9.9|14.930618340989255|
# MAGIC | 11.11|16.766099309517024|
# MAGIC | 22.22| 33.61915183872655|
# MAGIC |123.45| 187.1776133627482|
# MAGIC +------+------------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Best selling days for sales data
# MAGIC
# MAGIC This task is in two parts, the first part is to calculate a reference result with static data.<br>
# MAGIC And the second part is to start preparing to do a similar calculation with streaming data.
# MAGIC
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) the file `exercises/ex5/static/superstore_sales.csv` contains sales data from a retailer. The data is based on a dataset from [https://www.kaggle.com/datasets/soumyashanker/superstore-us](https://www.kaggle.com/datasets/soumyashanker/superstore-us).
# MAGIC
# MAGIC #### Background on the streaming data simulation
# MAGIC
# MAGIC In this exercise, tasks 6-8, streaming data is simulated by copying CSV files from a source folder to a target folder. The target folder can thus be considered as streaming data, with a new file appearing after each file is copied.<br>
# MAGIC A helper function to handle the file copying is given in task 8 which you don't need to modify.<br>
# MAGIC The source folder from which the files are copied is in the shared container, and the target folder will be in [Students container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/students/etag/%220x8DE01A3A1A7F5AB%22/defaultId//publicAccessVal/None) where everyone has write permissions.
# MAGIC
# MAGIC #### The tasks
# MAGIC
# MAGIC Part 1: calculation with static data
# MAGIC
# MAGIC - Read the data from the CSV file into a data frame called `staticSalesDF`.
# MAGIC - Calculate the total sales for each day, and show the eight best selling days.
# MAGIC     - Each row has a sales record for a specific product.
# MAGIC     - The column `productPrice` contains the price for an individual product.
# MAGIC     - The column `productCount` contains the count for how many items were sold in the given sale.
# MAGIC
# MAGIC Part 2: setting up a streaming data frame for upcoming data
# MAGIC
# MAGIC - Create a streaming data frame for similar retailer sales data as was used in part 1. The streaming data frame should point to the folder in the students container, i.e., to the address given by `myStreamingFolder`.
# MAGIC
# MAGIC Hint: Spark cannot infer the schema of streaming data, so you have to give it explicitly. You can assume that the streaming data will have the same format as the static data used in part 1, except that there will not be any header rows in the streaming data.

# COMMAND ----------

staticSalesDF = (
    spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("sep", ";") 
        .csv("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex5/static/superstore_sales.csv")
)


# COMMAND ----------


staticBestDaysDF: DataFrame = (
    staticSalesDF
        .withColumn("saleTotal", col("productPrice") * col("productCount"))
        .groupBy("orderDate")
        .agg(spark_sum("saleTotal").alias("dailySales"))
        .orderBy(col("dailySales").desc())
        .limit(8)
)


# COMMAND ----------

# Output for part 1 with the static data
print("The best selling days with the static data:")
staticBestDaysDF.show()

# COMMAND ----------

# identifier for your target folder to separate your streaming test from the others running at the same time
# this should only contain alphanumeric characters or underscores
myStreamingIdentifier: str = "najmeh_test"  # for example: "my_very_unique_identifier" or "firstname_lastname"

# setup the address for your folder in the students container in the Azure storage
myStreamingFolder: str = f"abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex5/{myStreamingIdentifier}/"
# Ensure the folder exists and is empty in the Azure storage when creating the streaming data frame
dbutils.fs.mkdirs(myStreamingFolder)
removeFiles(myStreamingFolder)


streamingSalesDF = (
    spark.readStream
        .format("csv")
        .schema(staticSalesDF.schema)
        .option("header", False)
        .option("sep", ";")      
        .load(myStreamingFolder)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The best selling days with the static data:
# MAGIC +----------+------------------+
# MAGIC | orderDate|        totalSales|
# MAGIC +----------+------------------+
# MAGIC |2014-03-18|          52148.32|
# MAGIC |2014-09-08|           22175.9|
# MAGIC |2017-11-04|          19608.77|
# MAGIC |2016-11-25|19608.739999999998|
# MAGIC |2016-10-02|18580.819999999996|
# MAGIC |2017-10-22|          18317.99|
# MAGIC |2016-05-23|16754.809999999998|
# MAGIC |2015-09-17|           16601.4|
# MAGIC +----------+------------------+
# MAGIC ```
# MAGIC
# MAGIC Note that you cannot really test part 2 of this task before you have also done the tasks 6 and 7. I.e. there is no checkable output from part 2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Best-selling days with streaming data
# MAGIC
# MAGIC Find the best-selling days using the streaming data frame from task 5, part 2.<br>
# MAGIC Also, this time include the total number of individual items sold on each day in addition to the total sales in the result.
# MAGIC
# MAGIC Note that in this task with the streaming data you don't need to limit the result only to the best eight selling days like was done in task 5, part 1.

# COMMAND ----------


streamingBestDaysDF = (
    streamingSalesDF
        .withColumn("saleTotal", col("productPrice") * col("productCount"))
        .groupBy("orderDate")
        .agg(
            spark_sum("saleTotal").alias("totalPrice"),
            spark_sum("productCount").alias("totalCount")
        )
        .orderBy(col("totalPrice").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that you cannot really test this task before you have also done the task 7. I.e. there is no checkable output from this task.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Running a streaming query
# MAGIC
# MAGIC Test your streaming data solution from tasks 5 and 6 by creating and starting a streaming query.
# MAGIC
# MAGIC The cell below this one contains helper scripts for copying the files (to simulate streaming data).<br>
# MAGIC The processing time for the streaming query might depend on the number of active users in Databricks.<br>
# MAGIC You might need to adjust the defined delays to get the system working as intended.

# COMMAND ----------

# There can be delays with the streaming data frame processing. You can try to adjust these wait times if you want.
waitAfterFirstCopy: int = 15
normalTimeInterval: int = 10
postLoopWaitTime: int = 10

def showQueryState(streamingQueryName: str) -> None:
    spark \
        .sql(f"SELECT * from {streamingQueryName}") \
        .show(8)

def copyFiles(streamingQueryName: str, targetFolder: str) -> None:
    streamingSource: str = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex5/streaming"
    inputFileList = dbutils.fs.ls(streamingSource)

    for index, csvFile in enumerate(inputFileList):
        inputFilePath: str = csvFile.path
        inputFile: str = inputFilePath.split("/")[-1]
        outputFilePath: str = targetFolder + inputFile

        # copy file from the shared container to the students container
        dbutils.fs.cp(inputFilePath, outputFilePath)
        waitTime: int = waitAfterFirstCopy if index == 0 else normalTimeInterval
        print(f"Copied file {inputFile} ({index + 1}/{len(inputFileList)}) to {outputFilePath} - waiting for {waitTime} seconds")
        time.sleep(waitTime)

        # show the current state of the streaming query
        showQueryState(streamingQueryName)

    print(f"Waiting additional {postLoopWaitTime} seconds")
    time.sleep(postLoopWaitTime)

# COMMAND ----------

# remove all files from myStreamingFolder before starting the streaming query to have a fresh run each time
removeFiles(myStreamingFolder)

# set up a unique identifier for the streaming query defined below (should only contain alphanumeric characters or underscores)
streamQueryName: str = f"ex5_{myStreamingIdentifier}"


# start the streaming query with the memory format and the query name set to the value of `streamQueryName`
myStreamingQuery = (
    streamingBestDaysDF
        .writeStream
        .format("memory")
        .queryName(streamQueryName)
        .outputMode("complete")
        .start()
)



# call the helper function to copy files to the target folder to simulate a streaming data
# the helper function will also show the current state of the streaming query after each copied file
copyFiles(streamQueryName, myStreamingFolder)

# show the final state of the streaming query
print("Final state of the streaming query before stopping:")
showQueryState(streamQueryName)

# stop the streaming query to avoid the notebook running indefinitely
myStreamingQuery.stop()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC If all copied files were handled by the streaming query, the final state should match the output from task 5, part 1<br>
# MAGIC (the `orderDate` and the `totalPrice` columns apart from possible rounding errors).
# MAGIC
# MAGIC <p>The target folder is dependent on `myStreamingIdentifier` and should not match what is shown here.</p>
# MAGIC <br>
# MAGIC
# MAGIC ```text
# MAGIC Copied file 0001.csv (1/10) to abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex5/example_solution/0001.csv - waiting for 15 seconds
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2014-07-26|          10931.66|        11|
# MAGIC |2017-12-07| 9454.390000000001|         8|
# MAGIC |2016-05-07|            4199.9|        10|
# MAGIC |2017-09-10|3943.4700000000003|         8|
# MAGIC |2017-09-09|3848.2000000000003|        15|
# MAGIC |2017-07-14|           3737.88|        12|
# MAGIC |2015-05-22|3716.6500000000005|         7|
# MAGIC |2016-12-03|3501.5699999999997|        12|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC
# MAGIC Copied file 0002.csv (2/10) to abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex5/example_solution/0002.csv - waiting for 10 seconds
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2017-11-17|          11052.78|        15|
# MAGIC |2014-07-26|11041.289999999999|        18|
# MAGIC |2017-12-07| 9454.390000000001|         8|
# MAGIC |2016-04-16| 9153.849999999999|        13|
# MAGIC |2017-08-17|           8917.08|        21|
# MAGIC |2017-10-13| 7259.849999999999|        10|
# MAGIC |2015-01-28|           7162.74|        13|
# MAGIC |2015-12-15| 6498.539999999999|        11|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC
# MAGIC ...
# MAGIC ...
# MAGIC ...
# MAGIC
# MAGIC Copied file 0010.csv (10/10) to abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex5/example_solution/0010.csv - waiting for 10 seconds
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2014-03-18|          52148.32|        48|
# MAGIC |2014-09-08|           22175.9|       112|
# MAGIC |2017-11-04|          19608.77|        58|
# MAGIC |2016-11-25|19608.739999999998|        53|
# MAGIC |2016-10-02|18580.819999999992|        27|
# MAGIC |2017-10-22|18317.989999999994|        49|
# MAGIC |2016-05-23|16754.809999999998|        23|
# MAGIC |2015-09-17|           16601.4|        86|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC
# MAGIC Waiting additional 10 seconds
# MAGIC Final state of the streaming query before stopping:
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2014-03-18|          52148.32|        48|
# MAGIC |2014-09-08|           22175.9|       112|
# MAGIC |2017-11-04|          19608.77|        58|
# MAGIC |2016-11-25|19608.739999999998|        53|
# MAGIC |2016-10-02|18580.819999999992|        27|
# MAGIC |2017-10-22|18317.989999999994|        49|
# MAGIC |2016-05-23|16754.809999999998|        23|
# MAGIC |2015-09-17|           16601.4|        86|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC ```
# MAGIC
# MAGIC Note that in Databricks the `awaitTermination` function of the streaming query seems to leave the notebook running and active even when used with the timeout parameter.<br>
# MAGIC Similarly, the Databricks `display` does not automatically stop with streaming data frames.
# MAGIC
# MAGIC You are free to do your own tests, but just be sure to stop all cells in the notebook once you are done.<br>
# MAGIC Closing the browser tab or the entire browser will not stop a running streaming query!

# COMMAND ----------

myStreamingQuery.stop()

# COMMAND ----------

for s in spark.streams.active:
    print(s.name, s.id, s.status)
    s.stop()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - Theory question
# MAGIC
# MAGIC Using your own words, answer the following questions:
# MAGIC
# MAGIC 1. Spark provides both low-level APIs (RDDs) and high-level APIs (DataFrames and Datasets) for data processing.
# MAGIC     - When should you prefer using RDDs over DataFrames?
# MAGIC     - When should you prefer using DataFrames over RDDs?
# MAGIC 2. The last coding tasks in this exercise related to streaming data.
# MAGIC     - What are the differences between processing static data and processing streaming data?
# MAGIC     - Using Spark, are there any limitations with processing streaming data compared to static data? If there are, give some examples.
# MAGIC
# MAGIC Extensive answers are not required here.<br>
# MAGIC If your answers do not fit into one screen, you have likely written more than what was expected.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. We use RDDs when:
# MAGIC   - we are working with **unstructured** data
# MAGIC   - the task requires **custom logic** that DataFrames cannot easily express
# MAGIC   - we need **full control** over data operations
# MAGIC   - we are performing **complex transformations** that do not fit the DataFrame API
# MAGIC
# MAGIC In other hand DataFrame is useful:
# MAGIC   - we are working with **structured** or **semi-structured** data
# MAGIC   - **performance** is important (DataFrames are optimized by Spark)
# MAGIC   - the **schema is known**
# MAGIC   - we want to use Spark libraries like **MLlib**, which work better and more easily with DataFrames
# MAGIC
# MAGIC 2. Static data is stored already, has a **fixed size**, and is **processed only once**. The **results are final**.
# MAGIC
# MAGIC Streaming data arrives **continuously**, has **no fixed size**, and is processed in repeated micro-batches. The results are updated continuously. New data can still arrive later, even for old timestamps.This is normal in real systems because:
# MAGIC   - data may be delayed
# MAGIC   - network may be slow
# MAGIC   - systems may send batches
# MAGIC   - sensors may buffer events
# MAGIC   - database logs may flush late

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use of AI and collaboration
# MAGIC
# MAGIC Using AI and collaborating with other students is allowed when doing the weekly exercises.<br>
# MAGIC However, the AI use and collaboration should be documented.
# MAGIC
# MAGIC - Did you use AI tools while doing this exercise?
# MAGIC   - Did they help? And how did they help?
# MAGIC - Did you work with other students to complete the tasks?
# MAGIC   - Only extensive collaboration is expected to be reported. If you only got help for a couple of the tasks, you don't need to report it here.

# COMMAND ----------

# MAGIC %md
# MAGIC I worked alone, but I used AI to help me understand RDDs more clearly. 
