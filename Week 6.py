# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 Data-Intensive Programming, Exercise 6
# MAGIC
# MAGIC This exercise demonstrates different file formats (CSV, Parquet, Delta) and some of the operations that can be used with them.<br>
# MAGIC The exercise is in three parts.
# MAGIC
# MAGIC - Tasks 1-4 concern reading and writing operations with CSV and Parquet
# MAGIC - Tasks 5-7 introduces the Delta format
# MAGIC - Task 8 is a theory question related to file formats.
# MAGIC
# MAGIC This is the **Python** version, switch to the Scala version if you want to do the tasks in Scala.
# MAGIC
# MAGIC Each task has its own cell(s) for the code. Add your solutions to the cells. You are free to add more cells if you feel it is necessary.<br>
# MAGIC There are cells with test code and example output following most of the tasks that involve producing code.
# MAGIC
# MAGIC At the end of the notebook, there is a question regarding the use of AI or other collaboration when working the tasks.<br>
# MAGIC Please remember to answer the AI question. And finally, don't forget to submit your solutions to Moodle: [Weekly Exercise #6](https://moodle.tuni.fi/mod/assign/view.php?id=3870687)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Some resources that can help with the tasks in this exercise:
# MAGIC
# MAGIC - The [tutorial notebook](https://adb-7895492183558578.18.azuredatabricks.net/editor/notebooks/743402606902162) from our course
# MAGIC - Chapters 4 and 9 in [Learning Spark, 2nd Edition](https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
# MAGIC     - There are additional code examples in the related [GitHub repository](https://github.com/databricks/LearningSparkV2).
# MAGIC     - The book related notebooks can be imported to Databricks by choosing `import` in your workspace and using the URL<br> `https://github.com/databricks/LearningSparkV2/blob/master/notebooks/LearningSparkv2.dbc`
# MAGIC - [Apache Spark documentation](https://spark.apache.org/docs/3.5.6/sql-ref-functions.html) on all available functions that can be used on DataFrames.<br>
# MAGIC   The full [Spark Scala functions API listing](https://spark.apache.org/docs/3.5.6/api/scala/org/apache/spark/sql/functions$.html) for the functions package might have some additional functions listed that have not been updated in the documentation.
# MAGIC - [What is Delta Lake?](https://docs.databricks.com/en/delta/index.html), [Delta Lake tutorial](https://docs.databricks.com/en/delta/tutorial.html), and [Upsert into Delta Lake](https://docs.databricks.com/en/delta/merge.html#modify-all-unmatched-rows-using-merge) pages from Databricks documentation.
# MAGIC - The Delta Spark [Python DeltaTable](https://docs.delta.io/latest/api/python/spark/index.html) documentation.

# COMMAND ----------

# some imports that might be required in the tasks
from functools import reduce
from typing import List

from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import col, concat, lit, to_timestamp, concat, regexp_replace

# COMMAND ----------

# some helper functions used in this exercise

def sizeInKB(sizeInBytes: int) -> float:
    return round(sizeInBytes/1024, 2)
def sizeInMB(sizeInBytes: int) -> float:
    return round(sizeInBytes/1024/1024, 2)
def folderSizeInKB(filePath: str) -> float:
    return sizeInKB(sum(file_info.size for file_info in dbutils.fs.ls(filePath)))

# print the files and their sizes from the target path
def printStorage(path: str) -> None:
    def getStorageSize(currentPath: str) -> int:
        # using Databricks utilities to get the list of the files in the path
        fileInformation: list[com.databricks.backend.daemon.dbutils.FileInfo] = dbutils.fs.ls(currentPath)
        sizes = []
        for file_info in fileInformation:
            if file_info.isDir():
                sizes.append(getStorageSize(file_info.path))
            elif not file_info.name.startswith("_"):
                print(f"{sizeInMB(file_info.size)} MB --- {file_info.path}")
                sizes.append(file_info.size)
        return sum(sizes)

    sizeInBytes = getStorageSize(path)
    print(f"Total size: {sizeInMB(sizeInBytes)} MB")

# remove all files and folders from the target path
def cleanTargetFolder(path: str) -> None:
    dbutils.fs.rm(path, True)

# Print column types in a nice format
def printColumnTypes(inputDF: DataFrame) -> None:
    maxColumnLength: int = max(len(column) for column in inputDF.columns)
    for column_name, column_type in inputDF.dtypes:
        print(f"{column_name}   {' ' * (maxColumnLength - len(column_name))}{column_type}")

# Returns a limited sample of the input data frame
def getTestDF(
    inputDF: DataFrame,
    ids: list[str] = ["Z1", "Z2"],
    limitRows: int = 2,
    idColumn: str = "ID",
    nonOrigIdentifier: str = "_"
) -> DataFrame:
    origSample: DataFrame = inputDF \
        .filter(~F.col(idColumn).contains(nonOrigIdentifier)) \
        .limit(limitRows)
    extraSample: DataFrame = reduce(
        lambda df1, df2: df1.union(df2),
        [inputDF.filter(F.col(idColumn).endswith(id)).limit(limitRows) for id in ids]
     )

    return origSample.union(extraSample)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Read and write data in two formats
# MAGIC
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) the `exercises/ex6` folder contains data about car accidents in the USA. The same data is given in multiple formats, and it is a subset of the dataset in Kaggle: [https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
# MAGIC
# MAGIC **Part 1**
# MAGIC
# MAGIC Read the data into data frames from both CSV and Parquet source format. The CSV source uses `|` as the column separator and contains header rows.
# MAGIC
# MAGIC Code for displaying the data and information about the source files is already included.
# MAGIC
# MAGIC **Part 2**
# MAGIC
# MAGIC Write the data from `df_csv` and `df_parquet` to the [Students container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/students/etag/%220x8DE01A3A1A7F5AB%22/defaultId//publicAccessVal/None) in both CSV and Parquet formats.
# MAGIC
# MAGIC Note, since the data in both data frames is the same, you can to use either one as the source when writing it to a new folder (regardless of the target format).

# COMMAND ----------

# Fill in your name (or some other unique identifier). This will be used to identify your target folder for the exercise.
student_name: str = "Najmeh"

source_path: str = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/"
target_path: str = f"abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/{student_name}/"
data_name: str = "accidents"

# COMMAND ----------


source_csv_folder: str = source_path + f"{data_name}_csv"

# create and display the data from CSV source
csv_path = source_path + "accidents_csv"


df_csv = (
    spark.read
        .option("header", "true")
        .option("delimiter", "|").
        option("inferSchema", "true")
        .csv(csv_path)
)

# df_csv = (
#     df_csv
#     .withColumn("Start_Time", to_timestamp(col("Start_Time")))
#     .withColumn("End_Time",   to_timestamp(col("End_Time")))
# )

# display(df_csv)

# COMMAND ----------

# Typically, a some more suitable file format would be used, like Parquet.
# With Parquet column format is stored in the file itself, so there is no need for schema inference or explicit schema.
source_parquet_folder: str = source_path + f"{data_name}_parquet"

# create and display the data from Parquet source
parquet_path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/accidents_parquet"
df_parquet: DataFrame =  spark.read.parquet(parquet_path)

display(df_parquet)

# COMMAND ----------

# print the list of source files for the different file formats
print("CSV files:")
printStorage(source_csv_folder)

print("\nParquet files:")
printStorage(source_parquet_folder)

# COMMAND ----------

# The schemas for both data frames should be the same (as long as both datasets have been correctly loaded)
print("===== CSV types =====")
printColumnTypes(df_csv)
print("\n===== Parquet types =====")
printColumnTypes(df_parquet)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC (the same output for both displays, only the first few lines shown here when using `show(5, false)` to view the data frame):
# MAGIC
# MAGIC ```text
# MAGIC +---------+-------------------+-------------------+---------------------------------------------------------------------------------+---------+------+-----+-------------+
# MAGIC |ID       |Start_Time         |End_Time           |Description                                                                      |City     |County|State|Temperature_F|
# MAGIC +---------+-------------------+-------------------+---------------------------------------------------------------------------------+---------+------+-----+-------------+
# MAGIC |A-3558690|2016-01-14 20:18:33|2017-01-30 13:25:19|Closed at Fullerton Ave - Road closed due to accident. Roadwork. Lane blocked.   |Whitehall|Lehigh|PA   |31.0         |
# MAGIC |A-3558700|2016-01-14 20:18:33|2017-01-30 13:34:02|Closed at Fullerton Ave - Road closed due to accident. Roadwork. Lane blocked.   |Whitehall|Lehigh|PA   |31.0         |
# MAGIC |A-3558713|2016-01-14 20:18:33|2017-01-30 13:55:44|Closed at Fullerton Ave - Road closed due to accident. Roadwork. Open.           |Whitehall|Lehigh|PA   |31.0         |
# MAGIC |A-3572241|2016-01-14 20:18:33|2017-02-17 23:22:00|Closed at Fullerton Ave - Road closed due to accident. Roadwork. Lane blocked.   |Whitehall|Lehigh|PA   |31.0         |
# MAGIC |A-3572395|2016-01-14 20:18:33|2017-02-19 00:38:00|Closed at Fullerton Ave - Road closed due to accident. Roadwork. Traffic problem.|Whitehall|Lehigh|PA   |31.0         |
# MAGIC +---------+-------------------+-------------------+---------------------------------------------------------------------------------+---------+------+-----+-------------+
# MAGIC only showing top 5 rows
# MAGIC ```
# MAGIC
# MAGIC <p>and</p>
# MAGIC <br>
# MAGIC
# MAGIC ```text
# MAGIC CSV files:
# MAGIC 31.97 MB --- abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/accidents_csv/us_traffic_accidents.csv
# MAGIC Total size: 31.97 MB
# MAGIC
# MAGIC Parquet files:
# MAGIC 9.56 MB --- abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/accidents_parquet/us_traffic_accidents.parquet
# MAGIC Total size: 9.56 MB
# MAGIC ```
# MAGIC
# MAGIC <p>and</p>
# MAGIC <br>
# MAGIC
# MAGIC ```text
# MAGIC ===== CSV types =====
# MAGIC ID              string
# MAGIC Start_Time      timestamp
# MAGIC End_Time        timestamp
# MAGIC Description     string
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_F   double
# MAGIC
# MAGIC ===== Parquet types =====
# MAGIC ID              string
# MAGIC Start_Time      timestamp
# MAGIC End_Time        timestamp
# MAGIC Description     string
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_F   double
# MAGIC ```

# COMMAND ----------

# remove all previously written files from the target folder first
cleanTargetFolder(target_path)

# the target paths for both CSV and Parquet
target_file_csv: str = target_path + data_name + "_csv"
target_file_parquet: str = target_path + data_name + "_parquet"

# write the data from part 1 in CSV format to the path given by target_file_csv
(
    df_csv.write
    .option("header", "true")
    .mode("overwrite")
    .csv(target_file_csv)
)

# write the data from part 1 in Parquet format to the path given by target_file_parquet
(
    df_parquet.write
    .mode("overwrite")
    .parquet(target_file_parquet)
)

# COMMAND ----------

# Check the written files:
printStorage(target_file_csv)
printStorage(target_file_parquet)

# Both with CSV and Parquet, the data can be divided into multiple files depending on how many workers were doing the writing.
# If a single file is needed, you can force the output into a single file with "coalesce(1)" before the write command
# This will make the writing less efficient, especially for larger datasets. (and is not needed in this exercise)
# There are some additional small metadata files (_SUCCESS, _committed, _started, .crc) that are not displayed in this output and that can be ignored in this exercise.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC (note that the number of files and the exact filenames will be different for you):
# MAGIC
# MAGIC ```text
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00000-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25667-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00001-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25668-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00002-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25669-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00003-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25670-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00004-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25680-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00005-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25681-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00006-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25682-1-c000.csv
# MAGIC 0.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00007-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25683-1-c000.csv
# MAGIC Total size: 31.97 MB
# MAGIC 9.56 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_parquet/part-00000-tid-2003618186209843740-a6c13b8d-8e28-471d-b7aa-d61b9b6b03fe-25705-1-c000.snappy.parquet
# MAGIC Total size: 9.56 MB
# MAGIC ```
# MAGIC
# MAGIC In this case, it is likely that when using the CSV source, `df_csv`, there will be multiple written files, regardless of the target format.<br>
# MAGIC And, when using the Parquet source, `df_parquet`, there will be just one data file, regardless of the target format.<br>
# MAGIC This behaviour is not general, and will not be true with the other data, especially with larger datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Add new rows to storage
# MAGIC
# MAGIC - Create a new data frame based on the task 1 data that contains the `75` latest incidents (based on the starting time) in the city of `Los Angeles`.
# MAGIC     - Append a postfix `_Z1` to the IDs of these Los Angeles incidents, e.g., `A-3666323` should be replaced with `A-3666323_Z1`.
# MAGIC - Create a second new data frame based on the task 1 data that contains the `50` oldest incidents (based on the starting time) in the city of `Chicago`.
# MAGIC     - Append a postfix `_Z1` also to the IDs of these Chicago incidents, e.g., `A-3499003` should be replaced with `A-3499003_Z1`.
# MAGIC - Append the rows from both new data frames to the CSV storage and to the Parquet storage.<br>
# MAGIC   I.e., write the new rows in append mode in CSV format to folder given by `target_file_csv` and in Parquet format to folder given by `target_file_parquet`.
# MAGIC - Finally, read the data from the storages again to check that the appending was successful.

# COMMAND ----------

los_angeles_rows: int = 75
chicago_rows: int = 50

# New data frame with Los Angeles accidents that will be appended to the storage
df_new_rows_los_angeles: DataFrame = (
    df_csv.filter(col("City") == "Los Angeles")
          .orderBy(col("Start_Time").desc())
          .limit(los_angeles_rows)
          .withColumn("ID", concat(col("ID"), lit("_Z1")))
)

# New data frame with Chicago accidents that will be appended to the storage
df_new_rows_chicago: DataFrame = (
    df_csv.filter(col("City") == "Chicago")
          .orderBy(col("Start_Time").asc())
          .limit(chicago_rows)
          .withColumn("ID", concat(col("ID"), lit("_Z1")))
)

# COMMAND ----------

# print out the first 2 rows of the new data frames
df_new_rows_los_angeles.show(2)
df_new_rows_chicago.show(2)

# COMMAND ----------

# Append the new rows to CSV storage:
# important to consistently use the same header and column separator options when using CSV storage
(
    df_new_rows_los_angeles.union(df_new_rows_chicago)
    .write
    .option("header", "true")
    .option("delimiter", "|")
    .mode("append")
    .csv(target_file_csv)
)

# Append the new rows to Parquet storage:
(
    df_new_rows_los_angeles.union(df_new_rows_chicago)
    .write
    .mode("append")
    .parquet(target_file_parquet)
)


# COMMAND ----------

# Read the merged data from the CSV files to check that the new rows have been stored
df_new_csv: DataFrame =(
    spark.read
    .option("header", "true")
    .option("delimiter", "|")
    .csv(target_file_csv)
)

# Read the merged data from the Parquet files to check that the new rows have been stored
df_new_parquet: DataFrame = spark.read.parquet(target_file_parquet)

# COMMAND ----------

print(f"Old DF had {df_parquet.count()} rows and we are adding {los_angeles_rows + chicago_rows} rows ", end="")
print(f"=> we should have {df_parquet.count() + los_angeles_rows + chicago_rows} in the merged data.")
print("==========================================================")
print(f"Old     CSV DF had {df_csv.count()} rows and new DF has {df_new_csv.count()} rows.")
print(f"Old Parquet DF had {df_parquet.count()} rows and new DF has {df_new_parquet.count()} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+
# MAGIC |          ID|         Start_Time|           End_Time|         Description|       City|     County|State|Temperature_F|
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|2023-03-29 07:55:41|San Diego Fwy S -...|Los Angeles|Los Angeles|   CA|         49.0|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|2023-03-23 13:45:00|CA-134 W - Ventur...|Los Angeles|Los Angeles|   CA|         58.0|
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+
# MAGIC only showing top 2 rows
# MAGIC +------------+-------------------+-------------------+--------------------+-------+------+-----+-------------+
# MAGIC |          ID|         Start_Time|           End_Time|         Description|   City|County|State|Temperature_F|
# MAGIC +------------+-------------------+-------------------+--------------------+-------+------+-----+-------------+
# MAGIC |A-3499003_Z1|2016-06-22 16:10:21|2016-06-22 22:10:21|Between 63rd St/E...|Chicago|  Cook|   IL|         73.0|
# MAGIC |A-3501512_Z1|2016-06-30 00:35:46|2016-06-30 06:35:46|Closed between 87...|Chicago|  Cook|   IL|         54.9|
# MAGIC +------------+-------------------+-------------------+--------------------+-------+------+-----+-------------+
# MAGIC only showing top 2 rows
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```text
# MAGIC Old DF had 198082 rows and we are adding 125 rows => we should have 198207 in the merged data.
# MAGIC ==========================================================
# MAGIC Old     CSV DF had 198082 rows and new DF has 198207 rows.
# MAGIC Old Parquet DF had 198082 rows and new DF has 198207 rows.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Append modified rows
# MAGIC
# MAGIC In the previous task, appending new rows was successful because the new data had the same schema as the original data.<br>
# MAGIC In this task, we try to append data with a modified schema to the CSV and Parquet storages.
# MAGIC
# MAGIC - Create a new data frame based on all rows from `df_new_rows_los_angeles` and `df_new_rows_chicago` from task 2. The data frame should be modified in the following way:
# MAGIC     - The values in the `ID` column should have a postfix `_Z2` instead of `_Z1`. E.g., `A-3877306_Z1` should be replaced with `A-3877306_Z2`.
# MAGIC     - A new column `AddedColumn1` should be added with values `"prefix-CITY"` where `CITY` is replaced by the city of the incident.
# MAGIC     - A new column `AddedColumn2` should be added with a constant value `"New column"`.
# MAGIC     - The column `Temperature_F` should be renamed to `Temperature_C` and the Fahrenheit values should be transformed to Celsius values.
# MAGIC         - Example of the temperature transformation: `49.0 °F` = `(49.0 - 32) / 9 * 5 °C` = `9.4444 °C`
# MAGIC     - The column `Description` should be dropped.
# MAGIC - Then append these modified rows to both the CSV storage and the Parquet storage.

# COMMAND ----------


# A new data frame with modified rows
temp_c_expr = (col("Temperature_F") - lit(32)) * lit(5.0) / lit(9.0)
df_modified: DataFrame = (
    df_new_rows_los_angeles.union(df_new_rows_chicago)
        # Replace _Z1 with _Z2
        .withColumn("ID", regexp_replace(col("ID"), "_Z1$", "_Z2"))
        # Convert Fahrenheit → Celsius, rename column
        .withColumn("Temperature_C", temp_c_expr)
        # Add AddedColumn1 = prefix-CITY
        .withColumn("AddedColumn1", concat(lit("prefix-"), col("City")))
        # Add AddedColumn2 = constant string
        .withColumn("AddedColumn2", lit("New column"))
        # remove old column
        .drop("Temperature_F")              
        # Drop Description column
        .drop("Description")
)

# COMMAND ----------

# Check the new data frame:
print(f"Rows in the new data frame: {df_modified.count()}")
print("Schema for the new data frame:")
printColumnTypes(df_modified)
print("The first 3 rows:")
df_modified.limit(3).show()

# COMMAND ----------

# Append the new modified rows to CSV storage:
(
    df_modified.write
        .option("header", "true")
        .option("delimiter", "|")
        .mode("append")
        .csv(target_file_csv)
)

# Append the new modified rows to Parquet storage:
(
    df_modified.write
        .mode("append")
        .option("mergeSchema", "true")  
        .parquet(target_file_parquet)
)


# COMMAND ----------

# Check the written files at this point:
printStorage(target_file_csv)
printStorage(target_file_parquet)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC Rows in the new data frame: 125
# MAGIC Schema for the new data frame:
# MAGIC ID              string
# MAGIC Start_Time      timestamp
# MAGIC End_Time        timestamp
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_C   double
# MAGIC AddedColumn1    string
# MAGIC AddedColumn2    string
# MAGIC The first 3 rows:
# MAGIC +------------+-------------------+-------------------+-----------+-----------+-----+------------------+------------------+------------+
# MAGIC |          ID|         Start_Time|           End_Time|       City|     County|State|     Temperature_C|      AddedColumn1|AddedColumn2|
# MAGIC +------------+-------------------+-------------------+-----------+-----------+-----+------------------+------------------+------------+
# MAGIC |A-3666323_Z2|2023-03-29 05:48:30|2023-03-29 07:55:41|Los Angeles|Los Angeles|   CA| 9.444444444444445|prefix-Los Angeles|  New column|
# MAGIC |A-3657191_Z2|2023-03-23 11:37:30|2023-03-23 13:45:00|Los Angeles|Los Angeles|   CA|14.444444444444445|prefix-Los Angeles|  New column|
# MAGIC |A-3779912_Z2|2023-01-31 00:58:00|2023-01-31 02:16:34|Los Angeles|Los Angeles|   CA|  8.88888888888889|prefix-Los Angeles|  New column|
# MAGIC +------------+-------------------+-------------------+-----------+-----------+-----+------------------+------------------+------------+
# MAGIC ```
# MAGIC
# MAGIC <p>and</p>
# MAGIC <br>
# MAGIC
# MAGIC ```text
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00000-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25667-1-c000.csv
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00000-tid-655426603930780467-2327cf74-80cc-4e8b-8e83-d6414f491743-32794-1-c000.csv
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00000-tid-9177609568380723532-b820de56-a8bd-4fd9-8ab4-ab385d267185-32928-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00001-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25668-1-c000.csv
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00001-tid-655426603930780467-2327cf74-80cc-4e8b-8e83-d6414f491743-32795-1-c000.csv
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00001-tid-9177609568380723532-b820de56-a8bd-4fd9-8ab4-ab385d267185-32929-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00002-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25669-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00003-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25670-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00004-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25680-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00005-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25681-1-c000.csv
# MAGIC 4.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00006-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25682-1-c000.csv
# MAGIC 0.5 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_csv/part-00007-tid-5821063222123259465-a29612c0-3864-4e4a-9976-d9907098dcc1-25683-1-c000.csv
# MAGIC Total size: 32.01 MB
# MAGIC 9.56 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_parquet/part-00000-tid-2003618186209843740-a6c13b8d-8e28-471d-b7aa-d61b9b6b03fe-25705-1-c000.snappy.parquet
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_parquet/part-00000-tid-5721364932986573223-3614e110-b76d-4d3a-92be-e5d83e479be5-32930-1-c000.snappy.parquet
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_parquet/part-00000-tid-7502162324470477129-1cf4dd8a-6865-459e-8dc0-75c1d731efca-32796-1-c000.snappy.parquet
# MAGIC 0.0 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_parquet/part-00001-tid-5721364932986573223-3614e110-b76d-4d3a-92be-e5d83e479be5-32931-1-c000.snappy.parquet
# MAGIC 0.01 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_parquet/part-00001-tid-7502162324470477129-1cf4dd8a-6865-459e-8dc0-75c1d731efca-32797-1-c000.snappy.parquet
# MAGIC Total size: 9.58 MB
# MAGIC ```
# MAGIC
# MAGIC As before, the exact file names and the number of files are likely different.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Check the merged data
# MAGIC
# MAGIC In this task, we check the contents of the CSV and Parquet storages after the two data append operations, the new rows with the same schema in task 2, and the new rows with a modified schema in task 3.
# MAGIC
# MAGIC ##### Part 1:
# MAGIC
# MAGIC The task is to first write the code that loads the data from the storages again. And then run the given test code that shows the number of rows, columns, schema, and some sample rows.
# MAGIC
# MAGIC ##### Part 2:
# MAGIC
# MAGIC Finally, answer the questions in the final cell of this task.

# COMMAND ----------

# Read in the CSV data again from the CSV storage: target_file_csv
modified_csv_df: DataFrame = (
    spark.read
        .option("header", "true")          # first row is header
        .option("inferSchema", "true")     # let Spark detect data types
        .csv(target_file_csv)
)

# COMMAND ----------

# CSV should have been broken in some way
print("===== CSV storage =====")
print(f"The number of rows should be correct: {modified_csv_df.count()} (i.e., {df_csv.count()}+2*{los_angeles_rows + chicago_rows})")
print(f"However, the original data had {len(df_csv.columns)} columns, inserted data had {len(df_modified.columns)} columns. Afterwards we have {len(modified_csv_df.columns)} columns while we should have {len(df_csv.columns) + 3} distinct columns.")

# print the schema of the CSV data frame
printColumnTypes(modified_csv_df)

# show two example rows from each addition
getTestDF(modified_csv_df).show()

# COMMAND ----------

# Read in the Parquet data again from the Parquet storage: target_file_parquet
modified_parquet_df: DataFrame =spark.read.parquet(target_file_parquet)

# COMMAND ----------

# Parquet should also be broken in some way
print("===== Parquet storage =====")
print(f"The count for number of rows might be wrong: {df_parquet.count()} (should be: {df_parquet.count()}+2*{los_angeles_rows + chicago_rows})")
print(f"Actually all {df_parquet.count()+2*(los_angeles_rows + chicago_rows)} rows should be included but the 2 conflicting schemas can cause the count to be incorrect.")
print(f"The original data had {len(df_parquet.columns)} columns, inserted data had {len(df_modified.columns)} columns. Afterwards we have {len(modified_parquet_df.columns)} columns while we should have {len(df_parquet.columns) + 3} distinct columns.")

# Unlike the CSV case, the data types for the columns have not been affected. But some columns are just ignored.
printColumnTypes(modified_parquet_df.withColumn("Temperature_F", col("Temperature_F").cast("string")))

# show two example rows from each additionc_name
getTestDF(modified_parquet_df).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC For CSV:
# MAGIC
# MAGIC ```text
# MAGIC The number of rows should be correct: 198332 (i.e., 198082+2*125)
# MAGIC However, the original data had 8 columns, inserted data had 9 columns. Afterwards we have 8 columns while we should have 11 distinct columns.
# MAGIC ID              string
# MAGIC Start_Time      string
# MAGIC End_Time        string
# MAGIC Description     string
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_F   string
# MAGIC +------------+--------------------+--------------------+--------------------+-----------+-----------+------------------+------------------+
# MAGIC |          ID|          Start_Time|            End_Time|         Description|       City|     County|             State|     Temperature_F|
# MAGIC +------------+--------------------+--------------------+--------------------+-----------+-----------+------------------+------------------+
# MAGIC |   A-4327762|2022-07-09T02:26:...|2022-07-09T05:23:...|Road closed due t...|    Cochise|    Cochise|                AZ|              72.0|
# MAGIC |   A-4033546|2022-07-09T02:30:...|2022-07-09T04:00:...|Incident on I-376...| Pittsburgh|  Allegheny|                PA|              71.0|
# MAGIC |A-3666323_Z1|2023-03-29T05:48:...|2023-03-29T07:55:...|San Diego Fwy S -...|Los Angeles|Los Angeles|                CA|              49.0|
# MAGIC |A-3657191_Z1|2023-03-23T11:37:...|2023-03-23T13:45:...|CA-134 W - Ventur...|Los Angeles|Los Angeles|                CA|              58.0|
# MAGIC |A-3666323_Z2|2023-03-29T05:48:...|2023-03-29T07:55:...|         Los Angeles|Los Angeles|         CA| 9.444444444444445|prefix-Los Angeles|
# MAGIC |A-3657191_Z2|2023-03-23T11:37:...|2023-03-23T13:45:...|         Los Angeles|Los Angeles|         CA|14.444444444444445|prefix-Los Angeles|
# MAGIC +------------+--------------------+--------------------+--------------------+-----------+-----------+------------------+------------------+
# MAGIC ```
# MAGIC
# MAGIC and for Parquet (alternative 1):
# MAGIC
# MAGIC ```text
# MAGIC ===== Parquet storage =====
# MAGIC The count for number of rows might be wrong: 198082 (should be: 198082+2*125)
# MAGIC Actually all 198332 rows should be included but the 2 conflicting schemas can cause the count to be incorrect.
# MAGIC The original data had 8 columns, inserted data had 9 columns. Afterwards we have 8 columns while we should have 11 distinct columns.
# MAGIC ID              string
# MAGIC Start_Time      timestamptype
# MAGIC End_Time        timestamptype
# MAGIC Description     string
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_F   double
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+
# MAGIC |          ID|         Start_Time|           End_Time|         Description|       City|     County|State|Temperature_F|
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+
# MAGIC |   A-3558690|2016-01-14 20:18:33|2017-01-30 13:25:19|Closed at Fullert...|  Whitehall|     Lehigh|   PA|         31.0|
# MAGIC |   A-3558700|2016-01-14 20:18:33|2017-01-30 13:34:02|Closed at Fullert...|  Whitehall|     Lehigh|   PA|         31.0|
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|2023-03-29 07:55:41|San Diego Fwy S -...|Los Angeles|Los Angeles|   CA|         49.0|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|2023-03-23 13:45:00|CA-134 W - Ventur...|Los Angeles|Los Angeles|   CA|         58.0|
# MAGIC |A-3666323_Z2|2023-03-29 05:48:30|2023-03-29 07:55:41|                NULL|Los Angeles|Los Angeles|   CA|         NULL|
# MAGIC |A-3657191_Z2|2023-03-23 11:37:30|2023-03-23 13:45:00|                NULL|Los Angeles|Los Angeles|   CA|         NULL|
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+
# MAGIC ```
# MAGIC
# MAGIC Parquet (alternative 2):
# MAGIC
# MAGIC ```text
# MAGIC ===== Parquet storage =====
# MAGIC The count for number of rows might be wrong: 198082 (should be: 198082+2*125)
# MAGIC Actually all 198332 rows should be included but the 2 conflicting schemas can cause the count to be incorrect.
# MAGIC The original data had 8 columns, inserted data had 9 columns. Afterwards we have 9 columns while we should have 11 distinct columns.
# MAGIC ID              string
# MAGIC Start_Time      timestamp
# MAGIC End_Time        timestamp
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_C   double
# MAGIC AddedColumn1    string
# MAGIC AddedColumn2    string
# MAGIC +------------+-------------------+-------------------+-----------+-----------+-----+------------------+------------------+------------+
# MAGIC |          ID|         Start_Time|           End_Time|       City|     County|State|     Temperature_C|      AddedColumn1|AddedColumn2|
# MAGIC +------------+-------------------+-------------------+-----------+-----------+-----+------------------+------------------+------------+
# MAGIC |   A-3558690|2016-01-14 20:18:33|2017-01-30 13:25:19|  Whitehall|     Lehigh|   PA|              NULL|              NULL|        NULL|
# MAGIC |   A-3558700|2016-01-14 20:18:33|2017-01-30 13:34:02|  Whitehall|     Lehigh|   PA|              NULL|              NULL|        NULL|
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|2023-03-29 07:55:41|Los Angeles|Los Angeles|   CA|              NULL|              NULL|        NULL|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|2023-03-23 13:45:00|Los Angeles|Los Angeles|   CA|              NULL|              NULL|        NULL|
# MAGIC |A-3666323_Z2|2023-03-29 05:48:30|2023-03-29 07:55:41|Los Angeles|Los Angeles|   CA| 9.444444444444445|prefix-Los Angeles|  New column|
# MAGIC |A-3657191_Z2|2023-03-23 11:37:30|2023-03-23 13:45:00|Los Angeles|Los Angeles|   CA|14.444444444444445|prefix-Los Angeles|  New column|
# MAGIC +------------+-------------------+-------------------+-----------+-----------+-----+------------------+------------------+------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC - **Did you get similar output for the data in CSV storage? If not, what was the difference?**
# MAGIC     - No, the row count is correct, but the schema is wrong, only 8 columns instead of 11 and all columns are read as string.
# MAGIC
# MAGIC - **What is your explanation/guess for why the CSV seems broken and the schema cannot be inferred anymore?**
# MAGIC     - CSV doesn’t store any schema, it’s just text.
# MAGIC
# MAGIC - **Did you get similar output for the data in Parquet storage, and which of the 2 alternatives? If not, what was the difference?**
# MAGIC     - No, Parquet did not include all appended rows and did not expose all columns.
# MAGIC
# MAGIC - **What is your explanation/guess for why not all 11 distinct columns are included in the data frame in the Parquet case?**
# MAGIC     - Parquet does store schema, but per file, not for the whole folder. When Spark reads a directory of Parquet files by default it does not merge schemas from different files. It usually takes the schema from one file and uses that as the DataFrame schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Delta - Reading and writing data
# MAGIC
# MAGIC [Delta](https://docs.databricks.com/en/delta/index.html) tables are a storage format that are more advanced. They can be used somewhat like databases.
# MAGIC
# MAGIC This is not native in Spark, but open format, which is more and more commonly used.
# MAGIC
# MAGIC Delta is stricter with data. We cannot, for example, have whitespace in column names, as you can have in Parquet and CSV. However, in this exercise, the example data is given with column names where these additional requirements have already been fulfilled. And thus, you don't have to worry about them in this exercise.
# MAGIC
# MAGIC Delta technically looks more or less like Parquet with some additional metadata files.
# MAGIC
# MAGIC ###### The task
# MAGIC
# MAGIC - In this task, read the source data given in Delta format into a data frame.
# MAGIC - And then write a copy of the data into the Students container to allow modifications in the following tasks.

# COMMAND ----------

source_delta_folder: str = source_path + f"{data_name}_delta"

# Read the original data in Delta format to a data frame
df_delta: DataFrame =  (
    spark.read
        .format("delta")
        .load(source_delta_folder)
)

# COMMAND ----------

print("The original data from the shared container:")
print(f"== Number or rows: {df_delta.count()}")
print("== Columns:")
printColumnTypes(df_delta)
print("== Storage files:")
printStorage(source_delta_folder)

# COMMAND ----------

target_file_delta: str = target_path + data_name + "_delta"

# write the data from df_delta using the Delta format to the path given by target_file_delta
(
    df_delta.write
        .format("delta")
        .mode("overwrite")
        .save(target_file_delta)
)

# COMMAND ----------

# Check the written files:
print("== Target files:")
printStorage(target_file_delta)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The original data from the shared container:
# MAGIC == Number or rows: 198082
# MAGIC == Columns:
# MAGIC ID              string
# MAGIC Start_Time      timestamp
# MAGIC End_Time        timestamp
# MAGIC Description     string
# MAGIC City            string
# MAGIC County          string
# MAGIC State           string
# MAGIC Temperature_F   double
# MAGIC == Storage files:
# MAGIC 0.0 MB --- abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/accidents_delta/_delta_log/00000000000000000000.crc
# MAGIC 0.0 MB --- abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/accidents_delta/_delta_log/00000000000000000000.json
# MAGIC 9.56 MB --- abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex6/accidents_delta/part-00000-35a096d7-a0ee-439a-85e0-aa78e2935f39-c000.snappy.parquet
# MAGIC Total size: 9.56 MB
# MAGIC ```
# MAGIC
# MAGIC <p>and</p>
# MAGIC <br>
# MAGIC
# MAGIC ```text
# MAGIC == Target files:
# MAGIC 0.0 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_delta/_delta_log/00000000000000000000.crc
# MAGIC 0.0 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_delta/_delta_log/00000000000000000000.json
# MAGIC 9.56 MB --- abfss://students@tunics320f2025gen2.dfs.core.windows.net/ex6/special-unique-name/accidents_delta/part-00000-acd2e020-f263-4f94-8c47-74192c65e6bf-c000.snappy.parquet
# MAGIC Total size: 9.57 MB
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Delta - Appending data and checking the results
# MAGIC
# MAGIC - Append the new rows using the same schema, `df_new_rows_los_angeles` and `df_new_rows_chicago` from task 2, to the Delta storage.
# MAGIC - Append the new rows using the modified schema, `df_modified` from task 3, to the Delta storage.
# MAGIC - Then, read the merged data and study whether the result with Delta is correct without lost or invalid data.

# COMMAND ----------

# Append the new rows using the same schema, from df_new_rows_los_angeles and df_new_rows_chicago, to the Delta storage:
(
    df_new_rows_los_angeles
        .unionByName(df_new_rows_chicago)
        .write
        .format("delta")
        .mode("append")
        .save(target_file_delta)
)

delta_after_task2: DataFrame = (
    spark.read
         .format("delta")
         .load(target_file_delta)
)

# By default, Delta is similar to Parquet in that it assumes the data schema to stay the same. However, we can enable it to handle schema modifications.
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", True)


# Append the new rows using the modified schema, df_modified, to the Delta storage:
(
    df_modified
        .write
        .format("delta")
        .option("mergeSchema", "true")
        .mode("append")
        .save(target_file_delta)
)

# COMMAND ----------

# Read the merged data from Delta storage to check that the new rows have been stored
modified_delta_df: DataFrame = (
    spark.read
         .format("delta")
         .load(target_file_delta)
)

# COMMAND ----------

print(f"The number of rows should be correct: {modified_delta_df.count()} (i.e., {df_delta.count()}+2*{los_angeles_rows + chicago_rows})")
print(f"The original data had {len(df_delta.columns)} columns, inserted data had {len(df_modified.columns)} columns. Afterwards we have {len(modified_delta_df.columns)} columns which should match the expected {len(df_delta.columns) + 3} distinct columns.")
print("Delta should handle the merge perfectly. The columns which were not given values are available with NULL values.")

# show two example rows from each addition
getTestDF(modified_delta_df).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The number of rows should be correct: 198332 (i.e., 198082+2*125)
# MAGIC The original data had 8 columns, inserted data had 9 columns. Afterwards we have 11 columns which should match the expected 11 distinct columns.
# MAGIC Delta should handle the merge perfectly. The columns which were not given values are available with NULL values.
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+------------------+------------------+------------+
# MAGIC |          ID|         Start_Time|           End_Time|         Description|       City|     County|State|Temperature_F|     Temperature_C|      AddedColumn1|AddedColumn2|
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+------------------+------------------+------------+
# MAGIC |   A-3558690|2016-01-14 20:18:33|2017-01-30 13:25:19|Closed at Fullert...|  Whitehall|     Lehigh|   PA|         31.0|              NULL|              NULL|        NULL|
# MAGIC |   A-3558700|2016-01-14 20:18:33|2017-01-30 13:34:02|Closed at Fullert...|  Whitehall|     Lehigh|   PA|         31.0|              NULL|              NULL|        NULL|
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|2023-03-29 07:55:41|San Diego Fwy S -...|Los Angeles|Los Angeles|   CA|         49.0|              NULL|              NULL|        NULL|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|2023-03-23 13:45:00|CA-134 W - Ventur...|Los Angeles|Los Angeles|   CA|         58.0|              NULL|              NULL|        NULL|
# MAGIC |A-3666323_Z2|2023-03-29 05:48:30|2023-03-29 07:55:41|                NULL|Los Angeles|Los Angeles|   CA|         NULL| 9.444444444444445|prefix-Los Angeles|  New column|
# MAGIC |A-3657191_Z2|2023-03-23 11:37:30|2023-03-23 13:45:00|                NULL|Los Angeles|Los Angeles|   CA|         NULL|14.444444444444445|prefix-Los Angeles|  New column|
# MAGIC +------------+-------------------+-------------------+--------------------+-----------+-----------+-----+-------------+------------------+------------------+------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Delta - Full modifications
# MAGIC
# MAGIC With CSV or Parquet, editing existing values is not possible without overwriting the entire dataset.
# MAGIC
# MAGIC Previously, we only added new lines. This way of working only supports adding new data. It does **not** support modifying existing data: updating values or deleting rows.<br>
# MAGIC (We could do that manually by adding a primary key and timestamp and always searching for the newest value.)
# MAGIC
# MAGIC Nevertheless, Delta tables take care of this and many more itself.
# MAGIC
# MAGIC This task is divided into four parts, with separate instructions for each cell. The first three parts ask for some code, and the final part is to run the given test code.
# MAGIC
# MAGIC **Part 1:** In the following cell, add the code to write the `df_delta_small` into Delta storage.

# COMMAND ----------

# Let us first save a smaller data so that it is easier to see what is happening
delta_table_file: str = target_path + data_name + "_deltatable_small"

# create a small 6 row data frame with only 5 columns
df_delta_small: DataFrame = getTestDF(modified_delta_df, ["Z1"], 3) \
    .drop("Description", "End_Time", "County", "AddedColumn1", "AddedColumn2")


# Write the new small data frame to storage in Delta format to path based on delta_table_file
(
    df_delta_small
        .write
        .format("delta")
        .mode("overwrite") 
        .save(delta_table_file)
)


# Create Delta table based on your target folder
deltatable: DeltaTable = DeltaTable.forPath(spark, delta_table_file)

# Show the data originally, before the merge that is done in the next part
print(f"== Originally, the size of Delta storage is {folderSizeInKB(delta_table_file)} kB and contains {deltatable.toDF().count()} rows.")
deltatable.toDF().sort(F.desc("Start_Time")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Delta tables are more like database type tables. We define them based on data and modify the table itself.
# MAGIC
# MAGIC We do this by telling Delta what is the primary key of the data. After this we tell it to "merge" new data to the old one. If primary key matches, we update the information. If primary key is new, add a row.
# MAGIC
# MAGIC **Part 2:** In the following cell, add the code to update the `deltatable` with the given updates in `df_delta_update`.<br>
# MAGIC The rows should be updated when the `ID` columns match. And if the id from the update is a new one, a new row should be inserted into the `deltatable`.

# COMMAND ----------

# create a 5 row data frame with the same columns with updated values for the temperature
df_delta_update: DataFrame = df_new_rows_los_angeles \
    .limit(5) \
    .drop("Description", "End_Time", "County") \
    .withColumn("Temperature_F", F.round(F.rand(seed=1) * 100, 1)) \


# code for updating the deltatable with df_delta_update
(
    deltatable.alias("t")
    .merge(
        df_delta_update.alias("u"),
        "t.ID = u.ID"    
    )
    .whenMatchedUpdateAll()   
    .whenNotMatchedInsertAll()
    .execute()
)


# Show the data after the merge
print(f"== After merge, the size of Delta storage is {folderSizeInKB(delta_table_file)} kB and contains {deltatable.toDF().count()} rows.")
deltatable.toDF().sort(F.desc("Start_Time")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Note: If you execute the previous cell multiple times, you can notice that the file size increases every time. However, the number of rows does not change.

# COMMAND ----------

# MAGIC %md
# MAGIC **Part 3:** As a second modification to the test data, do the following modifications to the `deltatable`:
# MAGIC
# MAGIC - Fill out the proper temperature values given in Celsius degrees for column `Temperature_C` for all incidents in the delta table.
# MAGIC - Remove all rows where the temperature in Celsius is below `-12 °C`.

# COMMAND ----------

# code for updating the deltatable with the Celsius temperature values
deltatable.update(
    condition="Temperature_F IS NOT NULL",
    set={
        "Temperature_C": F.round((F.col("Temperature_F") - F.lit(32)) * F.lit(5.0/9.0), 1)
    }
)

# code for removing rows where the temperature is below -12 Celsius degrees from the deltatable
deltatable.delete("Temperature_C < -12")

# Show the data after the second update
print(f"== After the second update, the size of Delta storage is {folderSizeInKB(delta_table_file)} kB and contains {deltatable.toDF().count()} rows.")
deltatable.toDF().sort(F.desc("Start_Time")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Part 4**: Run the following cell as a demonstration on how the additional and unused data can be removed from the storage.
# MAGIC
# MAGIC The modifications and removals are actually physically done only once something like the vacuuming shown below is done. Before that, the original data still exist in the original files.

# COMMAND ----------

# We can get rid of additional or unused data from the storage by vacuuming

# you can print the files before vacuuming by uncommenting the following
# printStorage(delta_table_file)

# Typically we do not want to vacuum all the data, only data older than 30 days or so.
# We need to tell Delta that we really want to do something stupid
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", False)
print(f"== Before vacuum, the size of Delta storage is {folderSizeInKB(delta_table_file)} kB.")
deltatable.vacuum(0)
print(f"== After vacuum, the size of Delta storage is {folderSizeInKB(delta_table_file)} kB.")

# you can print the files after vacuuming by uncommenting the following
# (there should be more metadata files (JSON and CRC), but the actual data (Parquet files) are truncated)
# printStorage(delta_table_file)

# the vacuuming should not change the actual data
deltatable.toDF().sort(F.desc("Start_Time")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC <p>The numbers for the files sizes might not match exactly.</p>
# MAGIC <br>
# MAGIC
# MAGIC ```text
# MAGIC == Originally, the size of Delta storage is 3.99 kB and contains 6 rows.
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |          ID|         Start_Time|       City|State|Temperature_F|Temperature_C|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|Los Angeles|   CA|         49.0|         NULL|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|Los Angeles|   CA|         58.0|         NULL|
# MAGIC |A-3779912_Z1|2023-01-31 00:58:00|Los Angeles|   CA|         48.0|         NULL|
# MAGIC |   A-3558690|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         NULL|
# MAGIC |   A-3558700|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         NULL|
# MAGIC |   A-3558713|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         NULL|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```
# MAGIC == After merge, the size of Delta storage is 6.19 kB and contains 8 rows.
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |          ID|         Start_Time|       City|State|Temperature_F|Temperature_C|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|Los Angeles|   CA|         63.6|         NULL|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|Los Angeles|   CA|         59.9|         NULL|
# MAGIC |A-3779912_Z1|2023-01-31 00:58:00|Los Angeles|   CA|         13.5|         NULL|
# MAGIC |A-5230341_Z1|2023-01-30 03:07:00|Los Angeles|   CA|          7.7|         NULL|
# MAGIC |A-4842824_Z1|2023-01-29 22:11:00|Los Angeles|   CA|         85.4|         NULL|
# MAGIC |   A-3558690|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         NULL|
# MAGIC |   A-3558700|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         NULL|
# MAGIC |   A-3558713|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         NULL|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```text
# MAGIC == After the second update, the size of Delta storage is 10.75 kB and contains 7 rows.
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |          ID|         Start_Time|       City|State|Temperature_F|Temperature_C|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|Los Angeles|   CA|         63.6|         17.6|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|Los Angeles|   CA|         59.9|         15.5|
# MAGIC |A-3779912_Z1|2023-01-31 00:58:00|Los Angeles|   CA|         13.5|        -10.3|
# MAGIC |A-4842824_Z1|2023-01-29 22:11:00|Los Angeles|   CA|         85.4|         29.7|
# MAGIC |   A-3558690|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         -0.6|
# MAGIC |   A-3558700|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         -0.6|
# MAGIC |   A-3558713|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         -0.6|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC ```
# MAGIC
# MAGIC and finally
# MAGIC
# MAGIC ```text
# MAGIC == Before vacuum, the size of Delta storage is 10.75 kB.
# MAGIC == After vacuum, the size of Delta storage is 4.4 kB.
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |          ID|         Start_Time|       City|State|Temperature_F|Temperature_C|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC |A-3666323_Z1|2023-03-29 05:48:30|Los Angeles|   CA|         63.6|         17.6|
# MAGIC |A-3657191_Z1|2023-03-23 11:37:30|Los Angeles|   CA|         59.9|         15.5|
# MAGIC |A-3779912_Z1|2023-01-31 00:58:00|Los Angeles|   CA|         13.5|        -10.3|
# MAGIC |A-4842824_Z1|2023-01-29 22:11:00|Los Angeles|   CA|         85.4|         29.7|
# MAGIC |   A-3558690|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         -0.6|
# MAGIC |   A-3558700|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         -0.6|
# MAGIC |   A-3558713|2016-01-14 20:18:33|  Whitehall|   PA|         31.0|         -0.6|
# MAGIC +------------+-------------------+-----------+-----+-------------+-------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - Theory question
# MAGIC
# MAGIC Using your own words, answer the following questions:
# MAGIC
# MAGIC 1. CSV and Parquet files have been used as the data sources in several exercises.
# MAGIC     - What benefits (if any) do you get when the data is stored in Parquet format instead of CSV format?
# MAGIC     - Are there drawbacks of using Parquet format over CSV format? When would you choose CSV over Parquet?
# MAGIC 2. This exercise considered data in CSV, Parquet, and Delta format.
# MAGIC     - What other file formats are supported by Spark?
# MAGIC     - Can you use other data sources than files with Spark?<br>
# MAGIC       If yes, give some source examples that can be used with Spark.
# MAGIC
# MAGIC Extensive answers are not required here.<br>
# MAGIC If your answers do not fit into one screen, you have likely written more than what was expected.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Parquet stores the schema and types with the data but CSV is just strings and needs schema inference or manual schemas every time.
# MAGIC Parquet is compressed, so it often takes much less space than raw text CSV. Also reading or filtering is faster than CSV. In other hand, Parquet is not human readable, we cant open it easily in editor and see the lines. Not all tools support it, but mostly support CSV.
# MAGIC When data set is small, or I need to work with handy tools or doing quick debugging I prefer working with CSV.
# MAGIC
# MAGIC 2. Other formats that Spark support: ORC, JSON, Avro, text and binary files.
# MAGIC
# MAGIC Spark can also read from many non-file sources:
# MAGIC - Databases: PostgreSQL, MySQL, Oracle
# MAGIC - Data warehouses: Snowflake, Redshift
# MAGIC - Cloud storage: S3, Azure Blob, Google Cloud Storage
# MAGIC - NoSQL: MongoDB, Cassandra, HBase
# MAGIC - Streaming: Kafka, Kinesis
# MAGIC - APIs: REST endpoints, JDBC/ODBC connections

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
# MAGIC I worked alone, but I used AI to help me debug question 4, since I had many problems getting the correct answer.