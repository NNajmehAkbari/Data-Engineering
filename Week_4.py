# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 Data-Intensive Programming, Exercise 4
# MAGIC
# MAGIC This exercise is in three parts.
# MAGIC
# MAGIC - Tasks 1-4 are basic tasks of using RDDs with text based data.
# MAGIC - Tasks 5-7 are similar basic tasks for text data but using DataFrames instead.
# MAGIC - Task 8 is a theory question related to the lectures.
# MAGIC
# MAGIC This is the **Python** version, switch to the Scala version if you want to do the tasks in Scala.
# MAGIC
# MAGIC Each task has its own cell(s) for the code. Add your solutions to the cells. You are free to add more cells if you feel it is necessary.<br>
# MAGIC There are cells with test code and example output following most of the tasks that involve producing code.
# MAGIC
# MAGIC At the end of the notebook, there is a question regarding the use of AI or other collaboration when working the tasks.<br>
# MAGIC Please remember to answer the AI question. And finally, don't forget to submit your solutions to Moodle: [Weekly Exercise #4](https://moodle.tuni.fi/mod/assign/view.php?id=3503819)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Some resources that can help with the tasks in this exercise:
# MAGIC
# MAGIC - The [tutorial notebook](https://adb-7895492183558578.18.azuredatabricks.net/editor/notebooks/743402606902162) from our course
# MAGIC - Chapters 3 and 6 in [Learning Spark, 2nd Edition](https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
# MAGIC     - There are additional code examples in the related [GitHub repository](https://github.com/databricks/LearningSparkV2).
# MAGIC     - The book related notebooks can be imported to Databricks by choosing `import` in your workspace and using the URL<br> `https://github.com/databricks/LearningSparkV2/blob/master/notebooks/LearningSparkv2.dbc`
# MAGIC - Apache Spark [RDD Programming Guide](https://spark.apache.org/docs/3.5.6/rdd-programming-guide.html)
# MAGIC - Apache Spark [Datasets and DataFrames](https://spark.apache.org/docs/3.5.6/sql-programming-guide.html#datasets-and-dataframes) documentation
# MAGIC - [Databricks tutorial](https://docs.databricks.com/en/getting-started/dataframes.html) of using Spark DataFrames
# MAGIC - [Apache Spark documentation](https://spark.apache.org/docs/3.5.6/sql-ref-functions.html) on all available functions that can be used on DataFrames.<br>
# MAGIC   The full [Spark Python functions API listing](https://spark.apache.org/docs/3.5.6/api/python/reference/pyspark.sql/functions.html) for the functions package might have some additional functions listed that have not been updated in the documentation.

# COMMAND ----------

# some imports that might be required in the tasks
import dataclasses
import re

from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Loading text data into an RDD
# MAGIC
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) the folder `exercises/ex4/wiki` contains texts from selected Wikipedia articles in raw text format.
# MAGIC
# MAGIC - Load all articles into a single RDD. Exclude all empty lines from the RDD.
# MAGIC - Count the total number of non-empty lines in the article collection.
# MAGIC - Pick the first 7 lines from the created RDD and print them out.

# COMMAND ----------

path = 'abfss://shared@tunics320f2025gen2.dfs.core.windows.net/exercises/ex4/wiki'
wikitextsRDD: RDD[str] = spark.read.text(path).rdd.map(lambda row: row['value']).filter(lambda line: line.strip() != '')

numberOfLines: int = wikitextsRDD.count()

lines7: list[str] = wikitextsRDD.take(7)

# COMMAND ----------

print(f"The number of lines with text: {numberOfLines}")
print("===================================")
# Print the first 100 characters of the first 7 lines
print(*[line[:100] for line in lines7], sep="\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The number of lines with text: 1113
# MAGIC ===================================
# MAGIC Artificial intelligence
# MAGIC Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particul
# MAGIC Some high-profile applications of AI include advanced web search engines (e.g., Google Search); reco
# MAGIC The various subfields of AI research are centered around particular goals and the use of particular
# MAGIC Artificial intelligence was founded as an academic discipline in 1956, and the field went through mu
# MAGIC Goals
# MAGIC The general problem of simulating (or creating) intelligence has been broken into subproblems. These
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Counting the number of words
# MAGIC
# MAGIC Using the RDD from task 1 as a starting point:
# MAGIC
# MAGIC - Create `wordsRdd` where each row contains one word with the following rules for words:
# MAGIC     - The word must have at least one character.
# MAGIC     - The word must not contain any numbers, i.e. the digits 0-9.
# MAGIC - Calculate the total number of words in the article collection using `wordsRDD`.
# MAGIC - Calculate the total number of distinct words in the article collection using the same criteria for the words.
# MAGIC
# MAGIC You can assume that words in the same line are separated from each other in the article collection by whitespace characters (` `).<br>
# MAGIC In this exercise you can ignore capitalization, parenthesis, and punctuation characters. I.e., `word`, `Word`, `WORD`, `word.`, `(word)`, and `word).` should all be considered as valid and distinct words for this exercise.

# COMMAND ----------

wordsRDD: RDD[str] = (
    wikitextsRDD
    .flatMap(lambda line: line.split(" "))
    .filter(lambda word: len(word) > 0)
    .filter(lambda word: not re.search(r'\d', word))
    )


numberOfWords: float = wordsRDD.count()

numberOfDistinctWords: float = wordsRDD.distinct().count()

# COMMAND ----------

print(f"The total number of words not containing digits: {numberOfWords}")
print(f"The total number of distinct words not containing digits: {numberOfDistinctWords}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The total number of words not containing digits: 44604
# MAGIC The total number of distinct words not containing digits: 9463
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Counting word occurrences with RDD
# MAGIC
# MAGIC - Using the article collection data, create a pair RDD, `wordCountRDD`, where each row contains a distinct word and the count for how many times the word can be found in the collection.
# MAGIC     - Use the same word criteria as in task 2, i.e., ignore words which contain digits.
# MAGIC
# MAGIC - Using the created pair RDD, find what is the most common 8-letter word that is not `software` and starts with an `s`, and what is its count in the collection.
# MAGIC     - You can modify the given code and find the word and its count separately if that seems easier for you.

# COMMAND ----------

#wordCountRDD: RDD[tuple[str, int]] = ???

#askedWord, wordCount = ???

import re

wordCountRDD = (
    wikitextsRDD
    .flatMap(lambda line: line.split())
    .filter(lambda word: len(word) > 0)
    .filter(lambda word: not re.search(r'\d', word))
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: (x[1], x[0]), ascending=True)
)

askedWord, wordCount = (
    wordCountRDD
    .filter(lambda x: len(x[0]) == 8)
    .filter(lambda x: x[0].startswith('s'))
    .filter(lambda x: x[0] != 'software')
    .max(key=lambda x: x[1])
)

# COMMAND ----------

print(f"First row in wordCountRDD: word: '{wordCountRDD.first()[0]}', count: {wordCountRDD.first()[1]}")
print(f"The most common 8-letter word that is not 'software' and starts with 's': '{askedWord}' (appears {wordCount} times)")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC (the values for the first row depend on the code, and can be different than what is shown here)
# MAGIC
# MAGIC ```text
# MAGIC First row in wordCountRDD: word: 'sense,', count: 1
# MAGIC The most common 8-letter word that is not 'software' and starts with 's': 'specific' (appears 41 times)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Digit occurrences in text
# MAGIC
# MAGIC Using the article collection data, create a pair RDD, `digitsRDD`, where each row contains digits from 0 to 9, and the average number of times each digit appears in a line in the source data. Order the RDD such that the most common digit is given first.
# MAGIC
# MAGIC As before, ignore all those lines that are empty in the article collection in this task.

# COMMAND ----------

digitCountsRDD: RDD[tuple[int, float]] = (
    wikitextsRDD
    .flatMap(lambda line: re.findall(r'\d', line)) 
    .map(lambda d: (d, 1))  
    .reduceByKey(lambda a, b: a + b)  
)

digitsRDD = (
    digitCountsRDD
    .map(lambda x: (x[0], x[1] / numberOfLines))   
    .sortBy(lambda x: x[1], ascending=False)       
)

# COMMAND ----------

print("The average number of times the digits appear in the source data lines:")
for row in digitsRDD.collect():
    print(f"Digit '{row[0]}': => average: {row[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The average number of times the digits appear in the source data lines:
# MAGIC Digit '0': => average: 0.33513027852650495
# MAGIC Digit '2': => average: 0.2857142857142857
# MAGIC Digit '1': => average: 0.2776280323450135
# MAGIC Digit '9': => average: 0.18418688230008984
# MAGIC Digit '8': => average: 0.07457322551662174
# MAGIC Digit '6': => average: 0.0673854447439353
# MAGIC Digit '3': => average: 0.0637915543575921
# MAGIC Digit '4': => average: 0.05840071877807727
# MAGIC Digit '7': => average: 0.05660377358490566
# MAGIC Digit '5': => average: 0.05390835579514825
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Handling text data with DataFrames
# MAGIC
# MAGIC In this task the same Wikipedia article collection as in the RDD tasks 1-4 is used.<br>
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) the folder `exercises/ex4/wiki` contains the texts from selected Wikipedia articles in raw text format.
# MAGIC
# MAGIC Scala supports Datasets, which are typed DataFrames, and the corresponding Scala task is about using them instead of RDDs to redo tasks 1 and 2.<br>
# MAGIC The Datasets are not supported by pyspark, thus this Python task is about using DataFrames instead.
# MAGIC
# MAGIC Part 1:
# MAGIC
# MAGIC - Do task 1 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.
# MAGIC     - Load all articles into a single `DataFrame`. Exclude all empty lines from the DataFrame.
# MAGIC     - Count the total number of non-empty lines in the article collection.
# MAGIC     - Pick the first 7 lines from the created DataFrame and print them out.
# MAGIC
# MAGIC Part 2:
# MAGIC
# MAGIC - Do task 2 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.
# MAGIC - Using the DataFrame from first part of this task as a starting point:
# MAGIC     - Create `wordsDF` where each row contains one word with the following rules for words:
# MAGIC         - The word must have at least one character.
# MAGIC         - The word must not contain any numbers, i.e. the digits 0-9.
# MAGIC     - Calculate the total number of words in the article collection using `wordsDF`.
# MAGIC     - Calculate the total number of distinct words in the article collection using the same criteria for the words.
# MAGIC
# MAGIC You can assume that words in the same line are separated from each other in the article collection by whitespace characters (` `).<br>
# MAGIC In this exercise you can ignore capitalization, parenthesis, and punctuation characters. I.e., `word`, `Word`, `word.`, and `(word)` should all be considered as valid and distinct words for this exercise.

# COMMAND ----------

from pyspark.sql.functions import col
wikitextsDF: DataFrame = spark.read.text(path)

linesInDF: int = wikitextsDF.filter(col("value") != "").count()

first7Lines: list[str] = [
    row["value"]
    for row in wikitextsDF.filter(col("value") != "").limit(7).collect()
]

# COMMAND ----------

print(f"The number of lines with text: {linesInDF}")
print("===================================")
# Print the first 100 characters of the first 7 lines
print(*[line[:100] for line in first7Lines], sep="\n")

# Use show() to show the first 7 lines
wikitextsDF.show(7)

# COMMAND ----------

from pyspark.sql.functions import split, explode, lower, regexp_replace, length
wordsDF: DataFrame = (
    wikitextsDF
    .select(explode(split(col("value"), " ")).alias("word"))
    .filter(length(col("word")) > 0)
    .filter(~col("word").rlike(r"\d"))
)

wordsInDF: int = wordsDF.count()

distinctWordsInDF: int = wordsDF.distinct().count()

# COMMAND ----------

print(f"The total number of words not containing digits: {wordsInDF}")
print(f"The total number of distinct words not containing digits: {distinctWordsInDF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The number of lines with text: 1113
# MAGIC ===================================
# MAGIC Artificial intelligence
# MAGIC Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particul
# MAGIC Some high-profile applications of AI include advanced web search engines (e.g., Google Search); reco
# MAGIC The various subfields of AI research are centered around particular goals and the use of particular
# MAGIC Artificial intelligence was founded as an academic discipline in 1956, and the field went through mu
# MAGIC Goals
# MAGIC The general problem of simulating (or creating) intelligence has been broken into subproblems. These
# MAGIC +--------------------+
# MAGIC |               value|
# MAGIC +--------------------+
# MAGIC |Artificial intell...|
# MAGIC |Artificial intell...|
# MAGIC |Some high-profile...|
# MAGIC |The various subfi...|
# MAGIC |Artificial intell...|
# MAGIC |               Goals|
# MAGIC |The general probl...|
# MAGIC +--------------------+
# MAGIC only showing top 7 rows
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```text
# MAGIC The total number of words not containing digits: 44604
# MAGIC The total number of distinct words not containing digits: 9463
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Counting word occurrences with DataFrame
# MAGIC
# MAGIC Do task 3 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.<br>
# MAGIC Also, this time you should use the given data class `WordCount` for the final result instead of `(str, int)` tuple like in task 3 with the RDD.
# MAGIC
# MAGIC - Using the article collection data, create a `DataFrame`, `wordCountDF`, where each row contains a distinct word and the count for how many times the word can be found in the collection.
# MAGIC     - Use the same word criteria as in task 2, i.e., ignore words which contain digits.
# MAGIC     - Have the column names in the DataFrame match the data class `WordCount`, i.e., `word` and `count`.
# MAGIC
# MAGIC - Using the created DataFrame, find what is the most common 8-letter word that is not `software` and starts with an `s`, and what is its count in the collection.
# MAGIC     - The straightforward way is to determine the word and its count together. However, you can determine the word and its count separately if you want.
# MAGIC     - Give the final result as an instance of the data class.

# COMMAND ----------

@dataclasses.dataclass(frozen=True)
class WordCount:
    word: str
    count: int

# COMMAND ----------

from pyspark.sql.functions import desc

wordCountDF = (
    spark.read.text(path)
    .filter(col("value") != "")
    .select(explode(split(col("value"), " ")).alias("word"))
    .filter(length(col("word")) > 0)
    .filter(~col("word").rlike(r'\d'))
    .groupBy("word")
    .count()
)

result_df = (
    wordCountDF
    .filter((length(col("word")) == 8) & 
            (col("word").startswith('s')) & 
            (col("word") != 'software'))
    .orderBy(desc("count"), "word") 
    .limit(1)
)

result_row = result_df.first()
askedWordCount = WordCount(
    word=result_row["word"] if result_row else "",
    count=result_row["count"] if result_row else 0
)


# COMMAND ----------

print(f"First row in wordCountDF: word: '{wordCountDF.first()['word']}', count: {wordCountDF.first()['count']}")
print(f"Type of askedWordCount: {type(askedWordCount)}")
print(f"The most common 8-letter word that is not 'software' and starts with 's': '{askedWordCount.word}' (appears {askedWordCount.count} times)")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC (the values for the first row depend on the code, and can be different than what is shown here)
# MAGIC
# MAGIC ```text
# MAGIC First row in wordCountDF: word: 'By', count: 12
# MAGIC Type of askedWordCount: <class '__main__.WordCount'>
# MAGIC The most common 8-letter word that is not 'software' and starts with 's': 'specific' (appears 41 times)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Digit occurrences in text with Datasets
# MAGIC
# MAGIC Do task 4 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.<br>
# MAGIC
# MAGIC - Using the article collection data, create a `DataFrame`, `digitDF`, where each row contains digits from 0 to 9, and the average number of times each digit appears in a line in the source data. Order the DataFrame such that the most common digit is given first.
# MAGIC - Collect the data from `digitDF` to a list of data class `DigitAverage` instances.
# MAGIC
# MAGIC As before, ignore all those lines that are empty in the article collection in this task.

# COMMAND ----------

@dataclasses.dataclass(frozen=True)
class DigitAverage:
    digit: int
    average: float

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace, split, explode, desc

non_empty_df = wikitextsDF.filter(col("value") != "")
numberOfLines = non_empty_df.count()

digitDF = (
    non_empty_df
    .select(
        explode(
            split(
                regexp_replace(col("value"), r"\D", ""), ""
            )
        ).alias("digit")
    )
    .filter(col("digit") != "")  # Filter out empty strings
    .groupBy("digit")
    .count()
    .withColumn("average", col("count") / numberOfLines)
    .select("digit", "average")
    .orderBy(desc("average"))
)

digitAverages = [
    DigitAverage(digit=int(row["digit"]), average=float(row["average"]))
    for row in digitDF.collect()
]



# COMMAND ----------

print("The average number of times the digits appear in the source data lines:")
for digitAverage in digitAverages:
    print(f"Digit '{digitAverage.digit}': => average: {digitAverage.average}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC The average number of times the digits appear in the source data lines:
# MAGIC Digit '0': => average: 0.33513027852650495
# MAGIC Digit '2': => average: 0.2857142857142857
# MAGIC Digit '1': => average: 0.2776280323450135
# MAGIC Digit '9': => average: 0.18418688230008984
# MAGIC Digit '8': => average: 0.07457322551662174
# MAGIC Digit '6': => average: 0.0673854447439353
# MAGIC Digit '3': => average: 0.0637915543575921
# MAGIC Digit '4': => average: 0.05840071877807727
# MAGIC Digit '7': => average: 0.05660377358490566
# MAGIC Digit '5': => average: 0.05390835579514825
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - Theory question
# MAGIC
# MAGIC Using your own words, answer the following questions:
# MAGIC
# MAGIC 1. DataFrames, RDDs, and Datasets all have operations that can be divided into transformations and actions.
# MAGIC    - How would you describe transformations? And how would you describe actions?
# MAGIC    - How do actions differ from transformations?
# MAGIC    - Give some examples on both types of operations.
# MAGIC 2. Lazy evaluation was mentioned in the lectures.
# MAGIC     - What does lazy evaluation mean? How does it differ compared to eager evaluation?
# MAGIC     - Why is understanding lazy evaluation important when using Spark?
# MAGIC
# MAGIC Extensive answers are not required here.<br>
# MAGIC If your answers do not fit into one screen, you have likely written more than what was expected.

# COMMAND ----------

# MAGIC %md
# MAGIC ***
# MAGIC ##### My Answer:
# MAGIC 1. Operations in Spark are divided into two groups: Transformations and Actions.
# MAGIC
# MAGIC     **- Transformations**: Think of them like route planning. They describe what to do but don’t actually run anything. Using a transformation alone doesn’t produce a result because Spark builds a plan instead of executing immediately. **groupBy, filter, map** are in this group.
# MAGIC
# MAGIC     **- Actions:** These actually execute the plan and return results or write data to storage. Same as **count, sum, collect**.
# MAGIC
# MAGIC     For example, calling groupBy alone does nothing until we apply an action like count() or sum().
# MAGIC
# MAGIC ***
# MAGIC 2. Lazy and eager evaluation are also related to transformations and actions.
# MAGIC
# MAGIC     **Lazy** means Spark doesn’t do anything until an action is called. **Eager** means operations are executed immediately.
# MAGIC
# MAGIC     It’s important to know this because it’s one of the reasons Spark is efficient compared to other systems — it doesn’t do any work unless it’s necessary.

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
# MAGIC I work alone, and I correct my mistakes using the AI provided on this page.