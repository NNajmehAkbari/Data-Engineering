# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 - Group assignment - Advanced task 2
# MAGIC
# MAGIC This is the **Python** version of the optional advanced task 2.<br>
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
# MAGIC     - This notebook is for **advanced task 2**
# MAGIC - Moodle submission for the group

# COMMAND ----------

# imports for the entire notebook
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import input_file_name
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
import re
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 2 - Wikipedia articles
# MAGIC
# MAGIC The folder `assignment/wikipedia` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains a number of the longest Wikipedia articles ([https://en.wikipedia.org/w/index.php?title=Special:LongPages&limit=500&offset=0](https://en.wikipedia.org/w/index.php?title=Special:LongPages&limit=500&offset=0)).<br>
# MAGIC The most recent versions of the articles were extracted in XML format using Wikipedia's export tool: [https://en.wikipedia.org/wiki/Special:Export](https://en.wikipedia.org/wiki/Special:Export)
# MAGIC
# MAGIC Spark has support for importing XML files directly to DataFrames, [https://spark.apache.org/docs/latest/sql-data-sources-xml.html](https://spark.apache.org/docs/latest/sql-data-sources-xml.html).
# MAGIC
# MAGIC #### Definition for a word to be considered in this task
# MAGIC
# MAGIC A word is to be considered (and included in the counts) in this task if<br>
# MAGIC
# MAGIC - when the following punctuation characters are removed: '`.`', '`,`', '`;`', '`:`', '`!`', '`?`', '`(`', '`)`', '`[`', '`]`', '`{`', '`}`',<br>
# MAGIC - and all letters have been changed to lower case, i.e., `A` -> `a`, ...
# MAGIC
# MAGIC the word fulfils the following conditions:
# MAGIC
# MAGIC - the word contains only letters in the English alphabet: '`a`', ..., '`z`'
# MAGIC - the word is at least 5 letters long
# MAGIC - the word is not the English word for a specific month:<br>
# MAGIC     `january`, `february`, `march`, `april`, `may`, `june`, `july`, `august`, `september`, `october`, `november`, `december`
# MAGIC - the word in not the English word for a specific season: `summer`, `autumn`, `winter`, `spring`
# MAGIC
# MAGIC For example, words `(These` and `country,` would be valid words to consider with these rules (as `these` and `country`).
# MAGIC
# MAGIC In this task, you can assume that each line in an article is separated by the new line character, '`\n`'.<br>
# MAGIC And that each word is separated by a whitespace character, '` `'.
# MAGIC
# MAGIC #### The tasks
# MAGIC
# MAGIC Load the content of the Wikipedia articles, and find the answers to the following questions using the presented criteria for a word:
# MAGIC
# MAGIC - What are the 10 most frequent words across all included articles?
# MAGIC     - Give the answer as a data frame with columns `word` and `total_count`.
# MAGIC - In which articles does the word `software` appear more than 5 times?
# MAGIC     - Give the answer as a list of articles titles.
# MAGIC - What are the 10 longest words that appear in at least 10% of the articles?
# MAGIC     - And the same for at least 25%, 50%, 75%, and 90% of the articles.
# MAGIC     - Words that have the same length should be ranked alphabetically.
# MAGIC     - Give the answer as a data frame with columns `rank`, `word_10`, `word_25`, `word_50`, `word_75`, and `word_90`.
# MAGIC - What are the 5 most frequent words per article in the articles last updated before October 2025?
# MAGIC     - In the answer, include the title and the date for the article, as well as the full character count for the article.
# MAGIC     - Give the answer as a data frame with columns `title`, `date`, `characters`, `word_1`, `word_2`, `word_3`, `word_4`, and `word_5`
# MAGIC         - where `word_1` would correspond to the most frequent word in the article, `word_2` the second most frequent word, ...
# MAGIC
# MAGIC Even though the tasks ask for data frame answers, RDDs or Datasets can be helpful. However, their use is optional, and all the tasks can be completed by only using data frames.

# COMMAND ----------

wikipediaDF = (
    spark.read.format("xml")
    .option("rowTag", "page")
    .option("recursiveFileLookup", "true")  # read all XML files recursively
    .load("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/wikipedia/")
)

# COMMAND ----------

wikipediaDF.show()

# COMMAND ----------

wikipediaDF.schema

# COMMAND ----------

# Extract the text content and title from the nested structure
articlesDF = wikipediaDF.select(
    F.col("title"),
    F.col("revision.text._VALUE").alias("text"),
    F.col("revision.timestamp").alias("timestamp")
)

# Define months and seasons to exclude
excluded_words = {
    'january', 'february', 'march', 'april', 'may', 'june', 
    'july', 'august', 'september', 'october', 'november', 'december',
    'summer', 'autumn', 'winter', 'spring'
}

# UDF to clean and filter words
def clean_and_filter_words(text):
    if text is None:
        return []
    
    # Define punctuation to remove
    punctuation = '.,:;!?()[]{}-'
    
    # Process text
    words = []
    for word in text.split():
        # Remove punctuation
        cleaned = ''.join(c for c in word if c not in punctuation)
        # Convert to lowercase
        cleaned = cleaned.lower()
        # Check conditions
        if (len(cleaned) >= 5 and 
            cleaned.isalpha() and 
            cleaned not in excluded_words):
            words.append(cleaned)
    
    return words

clean_words_udf = F.udf(clean_and_filter_words, ArrayType(StringType()))

# COMMAND ----------

# Apply the UDF to extract valid words
wordsDF = articlesDF.select(
    F.col("title"),
    F.col("timestamp"),
    F.col("text"),
    clean_words_udf(F.col("text")).alias("words")
)

# Explode the words array to get one row per word
explodedWordsDF = wordsDF.select(
    F.col("title"),
    F.col("timestamp"),
    F.col("text"),
    F.explode(F.col("words")).alias("word")
)

# Task 1: Count word frequencies across all articles
wordCountsDF = explodedWordsDF.groupBy("word").agg(
    F.count("*").alias("total_count")
).orderBy(F.desc("total_count"))

# COMMAND ----------

# Get top 10
tenMostFrequentWordsDF: DataFrame = wordCountsDF.limit(10)

# COMMAND ----------

print("Top 10 most frequent words across all articles:")
tenMostFrequentWordsDF.show(truncate=False)

# COMMAND ----------

# Task 2: Find articles where 'software' appears more than 5 times
softwareCountsDF = explodedWordsDF.filter(F.col("word") == "software") \
    .groupBy("title") \
    .agg(F.count("*").alias("software_count")) \
    .filter(F.col("software_count") > 5) \
    .orderBy("title")

# Convert to list of titles
softwareArticles: list[str] = [row.title for row in softwareCountsDF.collect()]

# COMMAND ----------

print("Articles in alphabetical order where the word 'software' appears more than 5 times:")
for title in softwareArticles:
    print(f" - {title}")

# COMMAND ----------

# Task 3: Find the 10 longest words that appear in at least X% of the articles

# First, get the total number of articles
total_articles = articlesDF.count()

# Get distinct articles per word
word_article_counts = explodedWordsDF.select("word", "title").distinct() \
    .groupBy("word") \
    .agg(F.count("title").alias("article_count"))

# Add word length
word_article_counts = word_article_counts.withColumn("word_length", F.length("word"))

# Calculate percentage thresholds
thresholds = {
    10: total_articles * 0.10,
    25: total_articles * 0.25,
    50: total_articles * 0.50,
    75: total_articles * 0.75,
    90: total_articles * 0.90
}

# Get top 10 longest words for each threshold
def get_top_words_for_threshold(threshold_pct, min_articles):
    return word_article_counts \
        .filter(F.col("article_count") >= min_articles) \
        .orderBy(F.desc("word_length"), F.col("word")) \
        .limit(10) \
        .select("word") \
        .rdd.flatMap(lambda x: x).collect()

# COMMAND ----------


# Get words for each threshold
words_10 = get_top_words_for_threshold(10, thresholds[10])
words_25 = get_top_words_for_threshold(25, thresholds[25])
words_50 = get_top_words_for_threshold(50, thresholds[50])
words_75 = get_top_words_for_threshold(75, thresholds[75])
words_90 = get_top_words_for_threshold(90, thresholds[90])

rows = []
for i in range(10):
    rows.append(Row(
        rank=i + 1,
        word_10=words_10[i] if i < len(words_10) else None,
        word_25=words_25[i] if i < len(words_25) else None,
        word_50=words_50[i] if i < len(words_50) else None,
        word_75=words_75[i] if i < len(words_75) else None,
        word_90=words_90[i] if i < len(words_90) else None
    ))

# COMMAND ----------

longestWordsDF: DataFrame = spark.createDataFrame(rows)

# COMMAND ----------

print("The longest words appearing in at least 10%, 25%, 50%, 75, and 90% of the articles:")
longestWordsDF.show(truncate=False)

# COMMAND ----------

# Task 4: Find the 5 most frequent words per article in articles last updated before October 2025
# Filter articles updated before October 2025
articlesBeforeOct2025 = wordsDF.filter(F.col("timestamp") < "2025-10-01")

# Explode words for these articles
explodedFiltered = articlesBeforeOct2025.select(
    F.col("title"),
    F.col("timestamp"),
    F.col("text"),
    F.explode(F.col("words")).alias("word")
)

# COMMAND ----------

# Count word frequencies per article
wordCountsPerArticle = explodedFiltered.groupBy("title", "word") \
    .agg(F.count("*").alias("word_count"))

# Rank words within each article
windowSpec = Window.partitionBy("title").orderBy(F.desc("word_count"), F.col("word"))
rankedWords = wordCountsPerArticle.withColumn("rank", F.row_number().over(windowSpec))

# Get top 5 words per article
top5PerArticle = rankedWords.filter(F.col("rank") <= 5)

# Pivot to get columns word_1, word_2, word_3, word_4, word_5
pivotedWords = top5PerArticle.groupBy("title").pivot("rank", [1, 2, 3, 4, 5]).agg(F.first("word"))

# Rename columns
pivotedWords = pivotedWords.select(
    F.col("title"),
    F.col("1").alias("word_1"),
    F.col("2").alias("word_2"),
    F.col("3").alias("word_3"),
    F.col("4").alias("word_4"),
    F.col("5").alias("word_5")
)

# COMMAND ----------

# Add date and character count from original articles
articlesInfo = articlesBeforeOct2025.select(
    F.col("title"),
    F.date_format(F.col("timestamp"), "yyyy-MM-dd").alias("date"),
    F.length(F.col("text")).alias("characters")
).distinct()

# Join with the pivoted words
frequentWordsDF = articlesInfo.join(pivotedWords, "title") \
    .select("title", "date", "characters", "word_1", "word_2", "word_3", "word_4", "word_5") \
    .orderBy("date", "title")

# COMMAND ----------

print("Top 5 most frequent words per article (excluding forbidden words) in articles last updated before October 2025:")
frequentWordsDF.show(truncate=False)
# display(frequentWordsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC # Extra (some visualization based on task 1)

# COMMAND ----------


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("="*80)
print("SECTION 1: WORD FREQUENCY PATTERNS")
print("="*80)

# Convert to pandas
top10_pd = tenMostFrequentWordsDF.toPandas()

# Calculate frequency distribution insights
total_top10 = top10_pd['total_count'].sum()
top10_pd['percentage'] = (top10_pd['total_count'] / total_top10 * 100).round(2)

print(f"\n Top 10 Word Statistics:")
print(f"   • Most frequent word: '{top10_pd.iloc[0]['word']}' ({top10_pd.iloc[0]['total_count']:,} times)")
print(f"   • Least frequent (in top 10): '{top10_pd.iloc[9]['word']}' ({top10_pd.iloc[9]['total_count']:,} times)")
print(f"   • Frequency ratio (1st/10th): {top10_pd.iloc[0]['total_count'] / top10_pd.iloc[9]['total_count']:.2f}x")

# Visualization 1: Top 10 words with percentages
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
bars = ax1.barh(range(len(top10_pd)), top10_pd['total_count'], color=plt.cm.viridis(np.linspace(0.3, 0.9, 10)))
ax1.set_yticks(range(len(top10_pd)))
ax1.set_yticklabels(top10_pd['word'])
ax1.invert_yaxis()
ax1.set_xlabel('Frequency Count', fontweight='bold', fontsize=11)
ax1.set_title('Top 10 Most Frequent Words', fontweight='bold', fontsize=13)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (count, pct) in enumerate(zip(top10_pd['total_count'], top10_pd['percentage'])):
    ax1.text(count, i, f'  {count:,} ({pct}%)', va='center', fontsize=9)

# Pie chart
ax2.pie(top10_pd['total_count'], labels=top10_pd['word'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Distribution Among Top 10 Words', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.show()