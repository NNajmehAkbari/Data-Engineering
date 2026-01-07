# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 - Group assignment - Advanced task 3
# MAGIC
# MAGIC This is the **Python** version of the optional advanced task 3.<br>
# MAGIC Switch to the Scala version, if you want to do the assignment in Scala.
# MAGIC
# MAGIC Add your solutions to the cells following the task instructions. You are free to add more cells if you feel it is necessary.<br>
# MAGIC The example outputs are given in a separate notebook in the same folder as this one.
# MAGIC
# MAGIC Look at the notebook for the basic tasks for general information about the group assignment.
# MAGIC
# MAGIC Don't forget to **submit your solutions to Moodle**, [Group assignment submission](https://moodle.tuni.fi/mod/assign/view.php?id=3503812), once your group is finished with the assignment.<br>
# MAGIC Moodle allows multiple submissions, so you can update your work after the initial submission until the deadline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Short summary on assignment points
# MAGIC
# MAGIC ##### Minimum requirements (points: 0-20 out of maximum of 60):
# MAGIC
# MAGIC - All basic tasks implemented (at least in "a close enough" manner)
# MAGIC - Moodle submission for the group
# MAGIC
# MAGIC ##### For those aiming for higher points (0-60):
# MAGIC
# MAGIC - All basic tasks implemented
# MAGIC - Correct and optimized solutions for the basic tasks (advanced task 1) (0-20 points)
# MAGIC - Two of the other three advanced tasks (2-4) implemented
# MAGIC     - Each graded advanced task will give 0-20 points
# MAGIC     - This notebook is for **advanced task 3**
# MAGIC - Moodle submission for the group

# COMMAND ----------

# imports for the entire notebook
import os
from pyspark.sql.functions import (
    col, 
    when, 
    concat_ws, 
    lit, 
    trim, 
    upper, 
    coalesce, 
    to_timestamp,
    regexp_extract, 
    regexp_replace,
    countDistinct, 
    sum, 
    count, 
    row_number, 
    desc
)
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

# COMMAND ----------

studentStoragePath: str = "abfss://students@tunics320f2025gen2.dfs.core.windows.net"

# COMMAND ----------

# returns a list of existing subdirectories under the input path
def getDirectoryList(path: str) -> list[str]:
    return sorted([
        file_info.path
        for file_info in dbutils.fs.ls(path)
        if file_info.isDir
    ])

# remove all files and folders from the target path
def cleanTargetFolder(path: str) -> None:
    dbutils.fs.rm(path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Phase 1 - Loading the data
# MAGIC
# MAGIC The folder `assignment/transactions` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains financial dataset with transaction records. The data is based on [https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) dataset which is made available under Apache License, Version 2.0, [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0). Only a limited part of the transaction data is included in this task.
# MAGIC
# MAGIC The dataset is divided into 24 parts, which have different file formats and can have slightly differing schemas.
# MAGIC
# MAGIC - The data in `Parquet` format is in the subdirectory `assignment/transactions/parquet`
# MAGIC - The data in `Apache ORC` format is in the subdirectory `assignment/transactions/orc`
# MAGIC - The data in `CSV` format is in the subdirectory `assignment/transactions/csv`
# MAGIC - The data in `JSON` format is in the subdirectory `assignment/transactions/json`
# MAGIC
# MAGIC You are given a helper function, `getDirectoryList`, that can be used to get the paths of the subdirectories under the input path.
# MAGIC
# MAGIC #### The task for phase 1
# MAGIC
# MAGIC - Load the data from all given parts and combine them together using the Delta Lake format. The goal is to write the combined data in Delta format to the [Students container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/students/etag/%220x8DE01A3A1A7F5AB%22/defaultId//publicAccessVal/None).

# COMMAND ----------

base_path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/transactions"

parquet_root = f"{base_path}/parquet"
orc_root = f"{base_path}/orc"
csv_root = f"{base_path}/csv"
json_root = f"{base_path}/json"


# COMMAND ----------

# The path for writing the data to the students container
targetPath: str = f"{studentStoragePath}/Najmeh_Team"

# this will remove all the files from the target path, i.e., a fresh start
cleanTargetFolder(targetPath)

# COMMAND ----------

def load_all_dirs(dir_root, format):
    # List subdirectories or files in the directory
    subs = dbutils.fs.ls(dir_root)
    dfs = []
    for sub in subs:
        # sub.path is a full path
        full_path = os.path.join(dir_root, sub.name)
        if format == "parquet":
            df = spark.read.parquet(full_path)
        elif format == "orc":
            df = spark.read.orc(full_path)
        elif format == "csv":
            df = spark.read.csv(full_path, header=True, inferSchema=True)
        elif format == "json":
            df = spark.read.json(full_path)
        else:
            continue
        dfs.append(df)
    return dfs

# COMMAND ----------

dfs_parquet = load_all_dirs(parquet_root, "parquet")
dfs_orc = load_all_dirs(orc_root, "orc")
dfs_csv = load_all_dirs(csv_root, "csv")
dfs_json = load_all_dirs(json_root, "json")

# all_dfs is a list of all DataFrames
all_dfs = dfs_parquet + dfs_orc + dfs_csv + dfs_json

# COMMAND ----------

def unify_schema(dfs):
    all_cols = sorted({col for df in dfs for col in df.columns})  
    # Align columns, if there are any missing columns, fill with null
    dfs_aligned = [df.select([col if col in df.columns else F.lit(None).alias(col) for col in all_cols]) for df in dfs]
    result = dfs_aligned[0]
    for df in dfs_aligned[1:]:
        result = result.unionByName(df, allowMissingColumns=True)
    return result

combinedDF = unify_schema(all_dfs)
combinedDF = combinedDF.select(
    "transaction_id",
    "timestamp",
    "client_id",
    "amount",
    "merchant_id",
    "merchant_city",
    "merchant_state",
    "merchant_country",
    "merchant",
    "amount_dollars",
    "date",
    "time"
)
combinedDF.write.format("delta").mode("overwrite").save(targetPath)

# COMMAND ----------

# test code for phase 1
transaction_ids: list[int] = [
    15471290, 15540933, 15614378, 15683708, 15743561, 15813630, 15887875, 15958050,
    16027329, 16097021, 16173489, 16243958, 16313703, 16384244, 16459459, 16529507,
    16605087, 16675317, 16745233, 16815275, 16890288, 16960180, 17030940, 17101718
]

phase1TestDF: DataFrame = spark.read.format("delta").load(targetPath)
print(f"Total number of transactions: {phase1TestDF.count()}")
print("Example transactions:")
phase1TestDF \
    .filter(F.col("transaction_id").isin(*transaction_ids)) \
    .orderBy(F.col("transaction_id").asc()) \
    .limit(24).show(24, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Phase 2 - Updating the data
# MAGIC
# MAGIC Four features regarding the transaction information are given in different ways in original data.
# MAGIC
# MAGIC - The timestamps are either given as TimestampType in `timestamp` column, or as StringTypes in `date` and `time` columns.
# MAGIC - The merchant ids are either given as IntegerType in `merchant_id` column, or as StringType in `merchant` column in a different format.
# MAGIC - The merchant state/country location is given in two different ways:
# MAGIC     - either as StringTypes in columns `merchant_state` and `merchang_country`
# MAGIC         - for locations outside the United States, the `merchant_state` will be an empty string in this case
# MAGIC     - or as a single StringType in column `merchant_state`
# MAGIC         - for locations in the United States, the column contains a 2-letter code for the state
# MAGIC         - for locations outside the United States, the column contains the name of the country
# MAGIC - The transaction amount is given either as StringType in `amount` column, or as DoubleType in `amount_dollars` column.
# MAGIC     - The string in the `amount` column is either in format `$64.63` or in format `64.63 USD`. Negative numbers are possible, e.g., `$-12.34` or `12.34 USD`.
# MAGIC
# MAGIC The folder `assignment/transactions/metadata/` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains a CSV file with the information on the US state names and their 2-letter abbreviations.
# MAGIC
# MAGIC #### The task for phase 2
# MAGIC
# MAGIC Update the combined data written in phase 1 with the following:
# MAGIC
# MAGIC - For rows which have the timestamp given with two columns, `date` and `time`, update the `timestamp` column value with the corresponding timestamp value.
# MAGIC - For rows which have the merchant id given as a string in column `merchant`, update the `merchant_id` column value with the corresponding integer value.
# MAGIC - For rows which have the merchant state/country location given in single column, `merchant_state`
# MAGIC     - update the `merchant_country` column with the corresponding country string
# MAGIC     - and update the `merchant_state` column with the full state name for US locations, and with an empty string for non-US locations
# MAGIC - For rows which have the transaction amount given as a string, update the `amount_dollars` column value with the corresponding double value.
# MAGIC
# MAGIC The goal is to have an updated dataset written in Delta format in the target location at the student container.<br>
# MAGIC And all the following columns should have non-null values: `transaction_id`, `timestamp`, `client_id`, `amount_dollars`, `merchant_id`, `merchant_city`, `merchant_state`, `merchant_country`

# COMMAND ----------

df = spark.read.format("delta").load(targetPath)
combinedDF = df

# 1. Fix merchant_id ------------------------------------------------------------------
combinedDF = combinedDF.withColumn(
    "merchant_id",
    F.when(
        col("merchant_id").isNull() & col("merchant").isNotNull(),
        regexp_extract(col("merchant"), r"merchant: (\d+)", 1).cast("int")
    ).otherwise(col("merchant_id"))
)

# 2. Fix timestamp ------------------------------------------------------------------
combinedDF = combinedDF.withColumn(
    "timestamp",
    F.when(
        col("timestamp").isNull() & col("date").isNotNull() & col("time").isNotNull(),
        to_timestamp(concat_ws(" ", col("date"), col("time")), "dd.MM.yyyy HH:mm")
    ).otherwise(col("timestamp"))
)

# 3. Fix amount_dollars ------------------------------------------------------------------
combinedDF = combinedDF.withColumn(
    "amount_dollars",
    F.when(
        col("amount_dollars").isNull() & col("amount").isNotNull(),
        regexp_replace(col("amount").cast("string"), r"[^0-9\.-]", "").cast("double")
    ).otherwise(col("amount_dollars"))
)

# 4. Fix merchant_state ------------------------------------------------------------------
states_path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/transactions/metadata/us_states.csv"
states_raw = spark.read.option("header", True).csv(states_path)
statesDF_codes = (states_raw
                  .select(trim(upper(col("abbreviation")))
                          .alias("state_key"), col("state")
                          .alias("state_full_name")))

statesDF_names = (states_raw
                  .select(trim(upper(col("state")))
                          .alias("state_key"), col("state")
                          .alias("state_full_name")))

statesDF_all = statesDF_codes.union(statesDF_names).dropDuplicates(["state_key"])


combinedDF = (combinedDF
              .withColumn("clean_state", trim(upper(col("merchant_state"))))
              .join(statesDF_all, col("clean_state") == statesDF_all.state_key, "left")
              )

is_us_state = col("state_full_name").isNotNull()

# Calculate new country and state values
new_country = (F
               .when(is_us_state, F.lit("United States"))
               .when(coalesce(col("merchant_country"), F.lit("")) == F.lit(""), col("merchant_state"))
               .otherwise(col("merchant_country")))


new_state = (F
             .when(is_us_state, col("state_full_name"))
             .when((~is_us_state) & (col("merchant_state") == new_country),F.lit(""))
             .otherwise(col("merchant_state")))

# Overwrite the columns ------------------------------------------------------------------
combinedDF = combinedDF.withColumn("merchant_country", new_country)
combinedDF = combinedDF.withColumn("merchant_state", new_state)

# Final select and write ------------------------------------------------------------------
final_cols = [
    "transaction_id", "timestamp", "client_id", "amount_dollars",
    "merchant_id", "merchant_city", "merchant_state", "merchant_country"
]
# Select the final required columns, ensuring they are non-null using coalesce
finalDF = combinedDF.select(
    *[coalesce(col(c).cast("long"), F.lit(-1)).alias(c) if c == "transaction_id" or c == "client_id"
      else coalesce(col(c).cast("int"), F.lit(-1)).alias(c) if c == "merchant_id"
      else coalesce(col(c).cast("double"), F.lit(0.0)).alias(c) if c == "amount_dollars"
      else coalesce(col(c).cast("timestamp"), F.lit("1970-01-01 00:00:00").cast("timestamp")).alias(c) if c == "timestamp"
      else coalesce(col(c).cast("string"), F.lit("")).alias(c)
      for c in final_cols]
)

# Write back to delta
finalDF.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(targetPath)

# COMMAND ----------

# test code for phase 2
phase2TestDF: DataFrame = spark.read.format("delta").load(targetPath) \
    .select("transaction_id", "timestamp", "client_id", "amount_dollars", "merchant_id", "merchant_city", "merchant_state", "merchant_country")

print(f"Total number of transactions: {phase2TestDF.count()}")
print("Example transactions:")
phase2TestDF \
    .filter(F.col("transaction_id").isin(*transaction_ids)) \
    .orderBy(F.col("transaction_id").asc()) \
    .show(24, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Phase 3 - Data calculations
# MAGIC
# MAGIC The transaction data is mostly US-based, but contains some transactions from other parts of the world.<br>
# MAGIC For this task, the top merchant is the one who had the largest total number of transactions.
# MAGIC
# MAGIC #### The task for phase 3
# MAGIC
# MAGIC Using the updated data from phase 2
# MAGIC
# MAGIC - Find the top 10 merchants selling in the United States with the following information:
# MAGIC     - number of total transactions
# MAGIC     - the number of US states the merchant has made transactions
# MAGIC     - the US state the merchant had the highest dollar total with the transactions
# MAGIC     - the total dollar amount for all transactions
# MAGIC - Find the top 10 merchants selling outside the United States with the following information:
# MAGIC     - number of total transactions
# MAGIC     - the number of countries the merchant has made transactions
# MAGIC     - the country the merchant had the highest dollar total with the transactions
# MAGIC     - the total dollar amount for all transactions
# MAGIC - Find the merchants that made just a single transaction in France in December 2015 with the following information:
# MAGIC     - the timestamp for the transaction
# MAGIC     - the client's id
# MAGIC     - the dollar amount for the transaction
# MAGIC     - the city the transaction was made in

# COMMAND ----------

targetPath = f"{studentStoragePath}/Najmeh_Team" 
phase3DF = spark.read.format("delta").load(targetPath)

# 1. Top 10 in the US ------------------------------------------------------------------
usDF = phase3DF.filter(trim(col("merchant_country")) == "United States")

us_state_agg = (usDF
                .groupBy("merchant_id", "merchant_state")
                .agg(sum("amount_dollars")
                     .alias("state_total_dollars")))

w_state = (Window
           .partitionBy("merchant_id")
           .orderBy(desc("state_total_dollars")))

us_best_state = (us_state_agg
                 .withColumn("rn", row_number()
                             .over(w_state)).filter(col("rn") == 1)
                 .select("merchant_id", col("merchant_state")
                         .alias("best_state")))

us_agg = usDF.groupBy("merchant_id").agg(
    count("*").alias("num_transactions"),
    countDistinct("merchant_state").alias("num_states"),
    sum("amount_dollars").alias("total_dollars")
)

usMerchantsDF = (us_agg
                 .join(us_best_state, "merchant_id", "left")
                 .orderBy(desc("total_dollars"))
                 .withColumn("total_dollars", F.round(col("total_dollars"), 2))
                 .select("merchant_id", "num_transactions", "num_states", "best_state", "total_dollars")
                 .limit(10))

# 2. Top 10  outside US ------------------------------------------------------------------
nonUSDF = (phase3DF
           .filter((trim(col("merchant_country")) != "United States") & (col("merchant_country")
                                                                         .isNotNull()) & (trim(col("merchant_country")) != "")))

nonus_country_agg = (nonUSDF
                     .groupBy("merchant_id", "merchant_country")
                     .agg(sum("amount_dollars")
                          .alias("country_total_dollars")))

w_country = Window.partitionBy("merchant_id").orderBy(desc("country_total_dollars"))

nonus_best_country = (nonus_country_agg
                      .withColumn("rn", row_number()
                                  .over(w_country)).filter(col("rn") == 1)
                      .select("merchant_id", col("merchant_country")
                              .alias("best_country")))

nonus_agg = nonUSDF.groupBy("merchant_id").agg(
    count("*").alias("num_transactions"),
    countDistinct("merchant_country").alias("num_countries"),
    sum("amount_dollars").alias("total_dollars")
)

nonUSMerchantsDF = (nonus_agg
                    .join(nonus_best_country, "merchant_id", "left")
                    .orderBy(desc("total_dollars"))
                    .withColumn("total_dollars", F.round(col("total_dollars"), 2))
                    .select("merchant_id", "num_transactions", "num_countries", "best_country", "total_dollars")
                    .limit(10))

# 3. Single transaction in France Dec 2015------------------------------------------------------------------
france_dec_2015 = phase3DF.filter(
    (trim(col("merchant_country")) == "France") &
    (col("timestamp") >= lit("2015-12-01 00:00:00")) &
    (col("timestamp") <  lit("2016-01-01 00:00:00"))
)

fr_counts = (france_dec_2015
             .groupBy("merchant_id")
             .agg(count("*").alias("tx_count"))
             .filter(col("tx_count") == 1)
             .select("merchant_id"))

franceMerchantsDF = (france_dec_2015
                     .join(fr_counts, "merchant_id", "inner")
                     .select("merchant_id", "timestamp", "client_id", col("amount_dollars")
                             .alias("dollars"), "merchant_city")
                     .orderBy("timestamp"))

# COMMAND ----------

print("Top 10 merchants selling in the US:")
usMerchantsDF.show(truncate=False)

# COMMAND ----------

print("Top 10 merchants selling outside the US:")
nonUSMerchantsDF.show(truncate=False)

# COMMAND ----------

print("The merchants having a single transaction in December 2015 in France:")
franceMerchantsDF.show(truncate=False)