# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 - Group assignment - Basic tasks
# MAGIC
# MAGIC This is the **Python** version of the assignment containing the compulsory basic tasks.<br>
# MAGIC Switch to the Scala version, if you want to do the assignment in Scala.
# MAGIC
# MAGIC In all tasks, add your solutions to the cells following the task instructions. You are free to add more cells if you feel it is necessary.<br>
# MAGIC The example outputs are given in a separate notebook in the same folder as this one.
# MAGIC
# MAGIC Don't forget to **submit your solutions to Moodle**, [Group assignment submission](https://moodle.tuni.fi/mod/assign/view.php?id=3503812), once your group is finished with the assignment.<br>
# MAGIC Moodle allows multiple submissions, so you can update your work after the initial submission until the deadline.
# MAGIC
# MAGIC ## Basic tasks (compulsory)
# MAGIC
# MAGIC There are, in total, eight basic tasks that every group must implement in order to have an accepted assignment.
# MAGIC
# MAGIC There are three separate datasets used in the coding tasks.
# MAGIC
# MAGIC - The basic task 1 deals with video game sales data.
# MAGIC - The basic tasks 2 and 3 use building location dataset.
# MAGIC - The basic tasks 4-7 are done using a dataset containing events from football matches.
# MAGIC - Finally, the basic task 8 asks some information on your assignment working process.
# MAGIC
# MAGIC ## Advanced tasks (optional)
# MAGIC
# MAGIC There are in total of four advanced tasks that can be done to gain some course points. Despite the name, the advanced tasks may or may not be harder than the basic tasks.
# MAGIC
# MAGIC The advanced task 1 asks you to do all the basic tasks in an optimized way. You might gain some points from this without directly trying, by just implementing the basic tasks efficiently. Logic errors and other issues that cause the basic tasks to give wrong results will be considered in the grading of the first advanced task. A maximum of 20 points will be given based on advanced task 1. Both the basic tasks and the advanced task 1 are done using this notebook, and submitted to Moodle together.
# MAGIC
# MAGIC The other three advanced tasks, are separate tasks and their implementation does not affect the grade given for the advanced task 1.<br>
# MAGIC Only two of the three available tasks will be graded, and each graded task can provide a maximum of 20 points to the total.<br>
# MAGIC You can attempt all three tasks, but only submit at most two of them to Moodle.<br>
# MAGIC Otherwise, the group assignment grader will randomly pick two of the tasks and ignore the third.
# MAGIC
# MAGIC - Advanced task 2 asks you to handle text articles extracted from Wikipedia.
# MAGIC - Advanced task 3 asks you to load in data from multiple formats, and then manipulate it using Delta format.
# MAGIC - Advanced task 4 asks you to do some classification related machine learning tasks with Spark.
# MAGIC
# MAGIC It is possible to gain partial points from the advanced tasks. I.e., if you have not completed the task fully but have implemented some part of the task, you might gain some appropriate portion of the points from the task. Logic errors, very inefficient solutions, and other issues will be taken into account in the task grading.
# MAGIC
# MAGIC The advanced tasks 2, 3, and 4 have separate notebooks that contain the actual tasks. The notebooks can be found in the same folder as this one. These advanced tasks are also submitted to Moodle as separate files.
# MAGIC
# MAGIC ## Assignment grading
# MAGIC
# MAGIC Failing to do the basic tasks, means failing the assignment and thus also failing the course!<br>
# MAGIC "A close enough" solutions might be accepted => even if you fail to do some parts of the basic tasks, submit your work to Moodle.
# MAGIC
# MAGIC Accepted assignment submissions will be graded from 0 to 60 points.
# MAGIC
# MAGIC The maximum grade that can be achieved by doing only the basic tasks is 20/60 points (through advanced task 1).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Short summary
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
# MAGIC - Moodle submission for the group

# COMMAND ----------

# add other required imports here
import re
import math
from functools import reduce
from typing import Dict, List, Tuple

# PySpark core
from pyspark.sql import DataFrame, SparkSession, types as T
from pyspark.sql.window import Window

# PySpark functions
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col,
    lit,
    when,
    year,
    to_date,
    round,
    concat_ws,
    array,
    array_contains,
    array_intersect,
    struct,
    desc,
    count,
    countDistinct,
    max as spark_max,
    sum as _sum,
    avg,
    min,
    cos,
    sin,
    acos,
    radians,
    year,
    broadcast,
    row_number
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 1 - Video game sales data
# MAGIC
# MAGIC The CSV file `assignment/sales/video_game_sales_2024.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains video game sales data.<br>
# MAGIC The data is based on [https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024) dataset which is made available under the ODC Attribution License, [https://opendatacommons.org/licenses/by/1-0/index.html](https://opendatacommons.org/licenses/by/1-0/index.html). The data used in this task includes only the video games for which at least some sales data is available, and some original columns have been removed.
# MAGIC
# MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
# MAGIC
# MAGIC Using the data, find answers to the following:
# MAGIC
# MAGIC - Which publisher has the highest total sales in video games in Japan, considering games released in years 2001-2010?
# MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2001-2010, what are the total sales, in Japan and globally, for each year? And how much of those global sales were for PlayStation 2 (PS2) games?
# MAGIC     - I.e., what are the total sales in Japan, in total globally, and in total for PS2 games, for video games released by this publisher in year 2001?<br>
# MAGIC       And the same for year 2002? ...
# MAGIC     - If some sales value is empty (i.e., NULL), it can be considered as 0 sales for that game in that region.

# COMMAND ----------

# 1. Loud csv
path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/sales/video_game_sales_2024.csv"
gameSales = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("delimiter", "|")
    .csv(path)
)


sales_cols = ["na_sales", "jp_sales", "pal_sales", "other_sales"]

for c in sales_cols:
    gameSales = gameSales.withColumn(c, col(c).cast("double"))
    gameSales = gameSales.na.fill({c: 0.0})

# Ensure release_date is a date type and extract year
gameSales = gameSales.withColumn(
    "release_year",
    year(
        to_date(col("release_date"), "yyyy-MM-dd")
    )
)

# COMMAND ----------

bestJapanPublisher: str = (
    gameSales
    .filter((col("release_year") >= 2001) & (col("release_year") <= 2010))
    .groupBy("publisher")
    .agg(_sum("jp_sales").alias("japan_sales"))
    .orderBy(col("japan_sales").desc())
)

top_publisher = bestJapanPublisher.first()


# Add a year column extracted from release_date
gameSales = gameSales.withColumn(
    "year",
    year(
        to_date(col("release_date"), "yyyy-MM-dd")
    )
)

# Add a global_sales column as the sum of all regional sales
gameSales = gameSales.withColumn(
    "global_sales",
    col("na_sales") + col("jp_sales") + col("pal_sales") + col("other_sales")
)

TOP = top_publisher["publisher"]

bestJapanPublisherSales: DataFrame = (
    gameSales
    .filter(
        (col("publisher") == TOP) &
        (col("year") >= 2001) &
        (col("year") <= 2010)
    )
    .groupBy("year")
    .agg(
        round(_sum("jp_sales"),2).alias("japan_total"),
        round(_sum("global_sales"),2).alias("global_total"),
        round(_sum(
            (col("global_sales") * (col("console") == "PS2").cast("int"))
        ),2).alias("ps2_total")
    )
    .orderBy("year")
)


# COMMAND ----------

print(f"The publisher with the highest total video game sales in Japan is: '{bestJapanPublisher}'")
print("Sales data for this publisher:")
bestJapanPublisherSales.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 2 - Building location data
# MAGIC
# MAGIC You are given a dataset containing the locations of buildings in Finland. The dataset is a subset from `https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f` (currently only available through the Wayback Machine: [https://web.archive.org/web/20241009075101/https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f](https://web.archive.org/web/20241009075101/https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f)) limited to only postal codes with the first two numbers in the interval 28-44 ([postal codes in Finland](https://www.posti.fi/en/zip-code-search/postal-codes-in-finland)).
# MAGIC
# MAGIC The dataset is in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) at folder `assignment/buildings`. The data is in Parquet format and the column names should be clear enough to understand the contents.
# MAGIC
# MAGIC Using the data, find the 10 municipalities that have the highest ratio of buildings per area within the municipality.
# MAGIC
# MAGIC - Each distinct postal code in the municipality is considered to be a separate area in this task.
# MAGIC - In your answer, include the following information about the 10 municipalities:
# MAGIC     - the municipality name
# MAGIC     - the number of areas within the municipality
# MAGIC     - the number of streets within the municipality
# MAGIC     - the number of buildings within the municipality
# MAGIC     - the building per area ratio
# MAGIC     - the shortest direct distance in kilometers between a building in the municipality and the kampusareena building at Hervanta campus
# MAGIC         - the building id for the kampusareena building is `101060573F`
# MAGIC         - you are given a haversine function that can be used to calculate the direct distance between two coordinate pairs

# COMMAND ----------

kampusareenaBuildingId: str = "101060573F"

# returns the distance between points (lat1, lon1) and (lat2, lon2) in kilometers
# based on https://community.esri.com/t5/coordinate-reference-systems-blog/distance-on-a-sphere-the-haversine-formula/ba-p/902128
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R: float = 6378.1  # radius of Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    deltaPhi = math.radians(lat2 - lat1)
    deltaLambda = math.radians(lon2 - lon1)

    a = (
        math.sin(deltaPhi * deltaPhi / 4.0) +
        math.cos(phi1) * math.cos(phi2) * math.sin(deltaLambda * deltaLambda / 4.0)
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# COMMAND ----------


path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/buildings"
buildingsDF = spark.read.parquet(path)

# Kampus Areena coordinates
kampus_row = (
    buildingsDF
        .filter(F.col("building_id") == kampusareenaBuildingId)
        .select("latitude_wgs84", "longitude_wgs84")
        .first()
)

kampus_lat = kampus_row["latitude_wgs84"]
kampus_lon = kampus_row["longitude_wgs84"]

# Haversine UDF
haversine_udf = F.udf(haversine, T.DoubleType())

# Add distance column
buildingsWithDistDF = (buildingsDF
                       .withColumn("distance_km", F
                                   .round(haversine_udf(F
                                                        .col("latitude_wgs84"), F
                                                        .col("longitude_wgs84"),F
                                                        .lit(kampus_lat),F
                                                        .lit(kampus_lon),
                ), 1
            )
        )
)



# COMMAND ----------

municipalityDF = (
    buildingsWithDistDF
        .groupBy("municipality")
        .agg(
            F.countDistinct("postal_code").alias("areas"),
            F.countDistinct("street").alias("streets"),
            F.countDistinct("building_id").alias("buildings"),
            F.min("distance_km").alias("min_distance"),
        )
        .withColumn(
            "buildings_per_area",
            F.round(F.col("buildings") / F.col("areas"), 1)
        )
        .select(
            "municipality",
            "areas",
            "streets",
            "buildings",
            "buildings_per_area",
            "min_distance",  
        )
        .orderBy(F.col("buildings_per_area").desc())
        .limit(10)
)


# COMMAND ----------

print("The 10 municipalities with the highest buildings per area (postal code) ratio:")
municipalityDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 3 - Finding address for building near average location
# MAGIC
# MAGIC Using the building location data from basic task 2, consider two subsets of buildings:
# MAGIC
# MAGIC 1. All the buildings in `Tampere`
# MAGIC 2. All the buildings within Hervanta area, i.e., buildings with a postal code of `33720`
# MAGIC
# MAGIC For both cases:
# MAGIC
# MAGIC - find the average coordinates using all the building locations in the subset
# MAGIC     - the latitude for the average coordinates is the average latitude for the buildings
# MAGIC     - the longitude for the average coordinates is the average longitude for the buildings
# MAGIC - find the address (i.e., street + house number) of the building that is closest to the average coordinates
# MAGIC - calculate the distance from the location of the closest building to the average coordinates

# COMMAND ----------

def hv_proxy(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2)

haversine_udf = F.udf(hv_proxy, T.DoubleType())

def closest_to_average(df):
    # Remove duplicate buildings by building_id
    df_unique = df.dropDuplicates(["building_id"])

    # Average latitude & longitude
    avg_row = df_unique.agg(
        F.avg("latitude_wgs84").alias("avg_lat"),
        F.avg("longitude_wgs84").alias("avg_lon")
    ).first()

    avg_lat = float(avg_row["avg_lat"])
    avg_lon = float(avg_row["avg_lon"])

    # Compute distance from each building to average
    df_with_dist = df_unique.withColumn(
        "dist_to_avg",
        haversine_udf(
            F.col("latitude_wgs84"),
            F.col("longitude_wgs84"),
            F.lit(avg_lat),
            F.lit(avg_lon)
        )
    )

    # Closest building
    closest = df_with_dist.orderBy("dist_to_avg").first()

    return {
        "avg_lat": avg_lat,
        "avg_lon": avg_lon,
        "closest_street": closest["street"],
        "closest_house_number": closest["house_number"],
        "distance_km": float(closest["dist_to_avg"])
    }
    

# COMMAND ----------

# postal code 33720
tampereDF = buildingsDF.filter(F.col("municipality") == "Tampere")
hervantaDF = buildingsDF.filter(F.col("postal_code") == 33720)

result_tampere = closest_to_average(tampereDF)
result_hervanta = closest_to_average(hervantaDF)


tampereAddress: str = f"{result_tampere['closest_street']} {result_tampere['closest_house_number']}"
hervantaAddress: str = f"{result_hervanta['closest_street']} {result_hervanta['closest_house_number']}"

# Use the built-in round for floats
tampereDistance: float = __builtins__.round(result_tampere['distance_km'], 3)
hervantaDistance: float = __builtins__.round(result_hervanta['distance_km'], 3)

# COMMAND ----------

print(f"The address closest to the average location in Tampere: '{tampereAddress}' at ({tampereDistance} km)")
print(f"The address closest to the average location in Hervanta: '{hervantaAddress}' at ({hervantaDistance} km)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 4 - Football data and the best goalscorers in Spain and Italy
# MAGIC
# MAGIC The folder `assignment/football/events` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1. The data is based on a dataset from [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5). The data is given in Parquet format.
# MAGIC
# MAGIC Additional player related information are given in Parquet format at folder `assignment/football/players`. This dataset contains information about the player names, default roles when playing, and their birth areas.
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium, and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
# MAGIC
# MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
# MAGIC
# MAGIC The team that scores more goals than their opponent wins the match.
# MAGIC
# MAGIC **Columns in the event data**
# MAGIC
# MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.<br>
# MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string | The name of the competition |
# MAGIC | season | string | The season the match was played |
# MAGIC | matchId | integer | A unique id for the match |
# MAGIC | eventId | integer | A unique id for the event |
# MAGIC | homeTeam | string | The name of the home team |
# MAGIC | awayTeam | string | The name of the away team |
# MAGIC | event | string | The main category for the event |
# MAGIC | subEvent | string | The subcategory for the event |
# MAGIC | eventTeam | string | The name of the team that initiated the event |
# MAGIC | eventPlayerId | integer | The id for the player who initiated the event, 0 for events not identified to a single player |
# MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
# MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
# MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
# MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC
# MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
# MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
# MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
# MAGIC
# MAGIC Note that there are two events related to each goal that happened in the matches covered by the dataset.
# MAGIC
# MAGIC - One event for the player who scored the goal. This includes possible own goals, i.e., accidentally directing the ball to their own goal.
# MAGIC - One event for the goalkeeper who tried to stop the goal.
# MAGIC
# MAGIC **Columns in the player data**
# MAGIC
# MAGIC Each row represents a single player. All the columns will not be needed in the assignment.
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerId     | integer     | A unique id for the player |
# MAGIC | firstName    | string      | The first name of the player |
# MAGIC | lastName     | string      | The last name of the player |
# MAGIC | birthArea    | string      | The birth area (nation or similar) of the player |
# MAGIC | role         | string      | The main role of the player, either `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
# MAGIC | foot         | string      | The stronger foot of the player |
# MAGIC
# MAGIC #### The task
# MAGIC
# MAGIC Using the given football data
# MAGIC
# MAGIC - Find the 7 players who scored the highest number of goals in `Spanish La Liga` during season `2017-2018`.
# MAGIC - Find the 7 players who scored the highest number of goals in `Italian Serie A` during season `2017-2018`.
# MAGIC
# MAGIC Give the results as DataFrames, which have one row for each player and the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | player         | string      | The name of the player (first name + last name) |
# MAGIC | team           | string      | The team that the player played for |
# MAGIC | goals          | integer     | The number of goals the player scored |
# MAGIC
# MAGIC In this task, you can assume that all the relevant players played for the same team for the entire season.

# COMMAND ----------

# Players + Events 
playerDF: DataFrame = spark.read.parquet("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/football/players/part-00000.parquet")

eventsDF : DataFrame = spark.read.parquet("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/football/events/*.parquet")
joinedDF = playerDF.join(eventsDF, playerDF.playerId == eventsDF.eventPlayerId, how="outer")

# COMMAND ----------

# Only scorer events (remove goalkeeper events)
goals = joinedDF.filter(
    ((joinedDF.event == "Shot")|(joinedDF.event == "Free Kick")) &
    (array_contains(joinedDF.tags, "Goal")|
    array_contains(joinedDF.tags,"Own goal")) &
    (~array_contains(joinedDF.tags, "Goalkeeper"))

)

# La Liga 2017–18
Laliga = goals.filter(
    (goals.competition == "Spanish La Liga") & (goals.season == "2017-2018")
)
laligaGoals = Laliga.groupBy(
    concat_ws(" ", Laliga.firstName, Laliga.lastName).alias("player"),
    Laliga.eventTeam.alias("club")
).count().withColumnRenamed("count", "goals")

goalscorersSpainDF = laligaGoals.orderBy(desc("goals")).limit(7)


# Serie A 2017–18
italy = goals.filter(
    (goals.competition == "Italian Serie A") & (goals.season == "2017-2018")
)
spanishGoals = italy.groupBy(
    concat_ws(" ", italy.firstName, italy.lastName).alias("player"),
    italy.eventTeam.alias("club")
).count().withColumnRenamed("count", "goals")
goalscorersItalyDF = spanishGoals.orderBy(desc("goals"), desc("club")).limit(7)


# COMMAND ----------

print("The top 7 goalscorers in Spanish La Liga in season 2017-18:")
goalscorersSpainDF.show(truncate=False)

# COMMAND ----------

print("The top 7 goalscorers in Italian Serie A in season 2017-18:")
goalscorersItalyDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 5 - Match appearances for Finnish players
# MAGIC
# MAGIC For this and the following task, a player is considered to have made an appearance in a match if,<br>
# MAGIC considering only events `Shot`, `Pass`, `Free Kick`, and `Save attempt`, the player has participated in at least 3 events in the match.
# MAGIC
# MAGIC Using the football data, find how many match appearances the Finnish players included in the player dataset made in season `2017-2018` considering all the available leagues.<br>
# MAGIC (for this task, the player is considered a Finnish player if their birth area is `Finland`)
# MAGIC
# MAGIC Give the results as a DataFrame, which have one row for each player and the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | player         | string      | The name of the player (first name + last name) |
# MAGIC | matches        | integer     | The number of matches the player made an appearance |

# COMMAND ----------


joinedDF = playerDF.join(eventsDF, playerDF.playerId == eventsDF.eventPlayerId, how="outer")

filteredDF = joinedDF.filter(
    (col("birthArea") == "Finland") &
    (
        (col("event") == "Shot") |
        (col("event") == "Free Kick") |
        (col("event") == "Pass") |
        (col("event") == "Save attempt")
    )
)

# Players name concatinated
playerCol = concat_ws(" ", col("firstName"), col("lastName")).alias("player")


# Count the number of events for each player in each match
eventsPerMatch = (
    filteredDF.groupBy(
        playerCol,
        col("matchId")
    )
    .count()
)

# At least 3 events in the match
appearances = eventsPerMatch.filter(col("count") >= 3)

# Count the number of matches for each player
resultDF = (
    appearances.groupBy("player")
    .agg(countDistinct("matchId").alias("matches"))
)

# Finnish players
finnPlayers = (
    joinedDF
    .filter(col("birthArea") == "Finland")
    .select(playerCol)
    .distinct()
)

# total appearances of Finnish players
finnishPlayersDF : DataFrame = (
    finnPlayers
    .join(resultDF, on="player", how="left")
    .na.fill(0, ["matches"])
    .orderBy(desc("matches"))
)

# COMMAND ----------

print("The number of matches the Finnish players made an appearance in:")
finnishPlayersDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 6 - Match appearances in multiple competitions
# MAGIC
# MAGIC In a single match, a player can naturally play for only one team. However, during the 2017-18 season several players were transferred or loaned to another team, and then made appearances for different teams and even in different competitions.
# MAGIC
# MAGIC Using the football data and the definition of a match appearance from basic task 5
# MAGIC
# MAGIC - Find the players who made at least 10 appearances in two separate competitions during season `2017-2018`.
# MAGIC
# MAGIC Give the results as a DataFrame, which have one row for each player and the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | player         | string      | The name of the player (first name + last name) |
# MAGIC | birthArea      | string      | The birth area (nation or similar) of the player |
# MAGIC | competition1   | string      | The name of the competition the player made the most appearances |
# MAGIC | matches1       | integer     | The number of competition1 matches the player made an appearance |
# MAGIC | competition2   | string      | The name of the competition the player made the second most appearances |
# MAGIC | matches2       | integer     | The number of competition2 matches the player made an appearance |
# MAGIC
# MAGIC For this task, you can assume that no player played matches in more than two competitions during season 2017-18.<br>
# MAGIC If the number of match appearances are equal, `competition1` should be the competition that is first in alphabetical order.

# COMMAND ----------


joinedDF = playerDF.join(eventsDF, playerDF.playerId == eventsDF.eventPlayerId, how="inner")

# Filter playesr during the 2017-2018 season
seasonDF = joinedDF.filter(col("season") == "2017-2018")

# Definition of a match appearance
events = ["Shot", "Pass", "Free Kick", "Save attempt"]
filteredSeasonDF = seasonDF.filter(col("event").isin(events))

# Count the number of events for each player in each match and each competition
eventsPerMatch = (
    filteredSeasonDF
        .groupBy(
            concat_ws(" ", col("firstName"), col("lastName")).alias("player"),
            col("matchId"),  
            col("competition")   
        )
        .count()
)

# At least 3 events in the match
appearances = eventsPerMatch.filter(col("count") >= 3)

# Count the number of matches for each player in each competition
appearancesPerCompetition = (
    appearances
        .groupBy(col("player"), col("competition"))
        .count()
        .withColumnRenamed("count", "matches")
)


# competition1 - most matches in competition
comp1 = (
    appearancesPerCompetition
    .groupBy("player")
    .agg(max(struct(col("matches"), col("competition"))).alias("max_struct"))
    .select(
        "player",
        col("max_struct.competition").alias("competition1"),
        col("max_struct.matches").alias("matches1")
    )
)

# most matches in competition2 
comp2 = (
    appearancesPerCompetition
    .join(comp1, on="player")
    .filter(col("competition") != col("competition1"))
    .groupBy("player")
    .agg(max(struct(col("matches"), col("competition"))).alias("second_struct"))
    .select(
        "player",
        col("second_struct.competition").alias("competition2"),
        col("second_struct.matches").alias("matches2")
    )
)


# Join the two DataFrames on the "player" column and select the desired columns
playerBirthArea = joinedDF.withColumn("player", concat_ws(" ", col("firstName"), col("lastName")))\
    .select("player", "birthArea").distinct()

appearanceDF: DataFrame = comp1.join(comp2, on="player", how="inner")\
    .join(playerBirthArea, on="player", how="left")\
    .select("player", "birthArea", "competition1", "matches1", "competition2", "matches2")\
    .filter((col("matches1") >= 10) & (col("matches2") >= 10))



# COMMAND ----------

print("Players who played in at least 10 matches in two separate competitions:")
appearanceDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 7 - Number of wins for teams
# MAGIC
# MAGIC Using the football data, calculate how many match wins each team achieved during season `2017-2018`.<br>
# MAGIC And then
# MAGIC
# MAGIC - Find out how many teams had at least 20 match wins during the season.
# MAGIC - For each competition, find the teams that had the most match wins in that competition.
# MAGIC
# MAGIC For the second part, give the results as a DataFrame, which have one row for each competition and the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | competition  | string      | The name of the competition |
# MAGIC | team         | string      | The name of the team|
# MAGIC | wins         | integer     | The number of match wins achieved by the team |
# MAGIC
# MAGIC You can assume that all teams achieved at least one match win during the season.

# COMMAND ----------

# Optional helper alias
spark_count = count

# Keep only necessary columns
eventsDF = eventsDF.select(
    "matchId", "season", "competition",
    "homeTeam", "awayTeam", "event",
    "eventTeam", "tags"
)

# Filter early (season + required events)
seasonDF = eventsDF.filter(
    (col("season") == "2017-2018") &
    col("competition").isNotNull()
)

# Separate regular goals and own goals
regularGoalsDF = seasonDF.filter(
    array_contains(col("tags"), "Goal") &
    (col("event") != "Save attempt")
).select("matchId", col("eventTeam").alias("scoringTeam"))

# For own goals, the OPPOSING team gets the goal
ownGoalsDF = seasonDF.filter(
    array_contains(col("tags"), "Own goal") &
    (col("event") != "Save attempt")
)

# Join with match info to determine the opposing team
matchInfoDF = seasonDF.select(
    "matchId", "homeTeam", "awayTeam"
).dropDuplicates(["matchId"])

ownGoalsWithOpponentDF = ownGoalsDF.alias("og").join(
    matchInfoDF.alias("m"), 
    col("og.matchId") == col("m.matchId")
).withColumn(
    "scoringTeam",
    when(col("og.eventTeam") == col("m.homeTeam"), col("m.awayTeam"))
    .otherwise(col("m.homeTeam"))
).select(col("og.matchId").alias("matchId"), "scoringTeam")

# Combine regular goals and own goals
allGoalsDF = regularGoalsDF.union(ownGoalsWithOpponentDF)

# Goals per (team, match)
goalsPerTeamDF = (
    allGoalsDF.groupBy("matchId", "scoringTeam")
              .agg(spark_count("*").alias("goals"))
)

# Unique match info
matchesDF = seasonDF.select(
    "matchId", "competition", "homeTeam", "awayTeam"
).dropDuplicates(["matchId"])

# Cache because we use it twice
matchesDF = matchesDF.cache()

# Add home/away goals - use aliases to avoid ambiguity
homeGoalsDF = (
    matchesDF.alias("m").join(
        goalsPerTeamDF.alias("g"),
        (col("m.matchId") == col("g.matchId")) &
        (col("m.homeTeam") == col("g.scoringTeam")),
        "left"
    ).select(
        col("m.matchId"),
        col("m.competition"),
        col("m.homeTeam"),
        col("m.awayTeam"),
        col("g.goals").alias("homeGoals")
    )
)

awayGoalsDF = (
    matchesDF.alias("m").join(
        goalsPerTeamDF.alias("g"),
        (col("m.matchId") == col("g.matchId")) &
        (col("m.awayTeam") == col("g.scoringTeam")),
        "left"
    ).select(
        col("m.matchId"),
        col("g.goals").alias("awayGoals")
    )
)

# Combine & fill nulls
matchesWithGoalsDF = (
    homeGoalsDF.join(awayGoalsDF, "matchId", "left")
               .fillna(0, ["homeGoals", "awayGoals"])
)

# Determine match winner
matchesWithWinnerDF = matchesWithGoalsDF.withColumn(
    "winner",
    when(col("homeGoals") > col("awayGoals"), col("homeTeam"))
    .when(col("awayGoals") > col("homeGoals"), col("awayTeam"))
)

# Count total wins per team
teamWinsDF = (
    matchesWithWinnerDF.filter(col("winner").isNotNull())
                       .groupBy("winner")
                       .agg(spark_count("*").alias("wins"))
                       .withColumnRenamed("winner", "team")
)

# Count teams with ≥ 20 wins
twentyWinsCount = teamWinsDF.filter(col("wins") >= 20).count()

# Wins per competition
teamWinsCompDF = (
    matchesWithWinnerDF.filter(col("winner").isNotNull())
                       .groupBy("competition", col("winner").alias("team"))
                       .agg(spark_count("*").alias("wins"))
)

# Best team per competition
bestTeamsDF = (
    teamWinsCompDF.groupBy("competition")
                  .agg(spark_max(struct(col("wins"), col("team"))).alias("max_struct"))
                  .select(
                      col("competition"),
                      col("max_struct.team").alias("team"),
                      col("max_struct.wins").alias("wins")
                  )
                  .orderBy(col("wins").desc())
)

# COMMAND ----------

print(f"Number of teams having at least 20 wins: {twentyWinsCount}")
print("The teams with the most wins in each competition:")
bestTeamsDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 8 - General information
# MAGIC
# MAGIC Answer the following questions.
# MAGIC
# MAGIC Remember that using AI and collaborating with other students outside your group is allowed as long as the usage and collaboration is documented.<br>
# MAGIC However, every member of the group should have some contribution to the assignment work.
# MAGIC
# MAGIC 1. Who were your group members and their contributions to the work?
# MAGIC     - Our group members are Najme Akbari, Arash Ghasemzadeh Kakroudi and Mahsa Boustani. We have had **serveral on-site meetings in the campus** in order to keep track of every tasks
# MAGIC     - We contributed in this way: 
# MAGIC         - Najme Akbari did the Basic Tasks 1,2,3 and also advanced task 3.
# MAGIC         - Arash Ghasemzadeh Kakroudi did the Advance Task 3 and partially collaborate with Mahsa in Basic Task 7.
# MAGIC         - Mahsa Boustani did Tasks 4,5,6,7 and partially collaborate with Najmeh in Advance Task 3.
# MAGIC         - For Advanced Task 1, all of us collaborated to do it.
# MAGIC
# MAGIC 	
# MAGIC
# MAGIC
# MAGIC 2. Did you use AI tools while doing the assignment?
# MAGIC     - We only used AI tools to help fix errors we encountered in Basic Task 7, Advanced Task 2, and Advanced Task 3. 
# MAGIC
# MAGIC 3. Did you work with students outside your assignment group?
# MAGIC     - No, we did not collaborate with any other students outside our tassignment group.

# COMMAND ----------

# MAGIC %md
# MAGIC # Advanced Task 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 1 - Optimized and correct solutions to the basic tasks
# MAGIC
# MAGIC - This advanced task 1 will be graded for every group with 0-20 course points.
# MAGIC
# MAGIC Use the tools Spark offers effectively and avoid unnecessary operations in the code for the basic tasks.
# MAGIC
# MAGIC A couple of things to consider (**not** a complete list):
# MAGIC
# MAGIC - Consider using explicit schemas when dealing with CSV data sources.
# MAGIC - Consider only including those columns from a data source that are actually needed.
# MAGIC - Filter unnecessary rows whenever possible to get smaller datasets.
# MAGIC - Avoid collect or similar expensive operations for large datasets.
# MAGIC - Consider using explicit caching if some data frame is used repeatedly.
# MAGIC - Avoid unnecessary shuffling (for example, grouping, joining, or sorting) operations.
# MAGIC - Avoid unnecessary actions (count, show, etc.) that are not needed for the task.
# MAGIC
# MAGIC In addition to the effectiveness of your solutions, the correctness of the solution logic will be considered when determining the grade for advanced task 1.
# MAGIC "A close enough" solution with some logic fails might be enough to have an accepted group assignment, but those failings are likely to lower the score for this task. Errors that prevent the grader for running your code without modifications, can be severely penalized.
# MAGIC
# MAGIC It is okay to have your own test code that would fall into category of "ineffective usage" or "unnecessary operations" while doing the assignment tasks. However, for the final Moodle submission you should comment out or delete such code (and test that you have not broken anything when doing the final modifications).
# MAGIC
# MAGIC Note, that you should not do the basic tasks again for this advanced task, but instead modify your basic task code with more efficient versions.
# MAGIC
# MAGIC You are encouraged to create a text cell below this one and describe what optimizations you have done.<br>
# MAGIC This might help the grader to better recognize how skilled your work with the basic tasks has been.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 1
# MAGIC
# MAGIC In this part, several points were considered for improvement:
# MAGIC
# MAGIC - We initially used inferSchema and then converted the release_date column again to be sure about its type. This approach is unnecessary. We now check the schema first before doing any conversion. Since the data was already in the correct format, we avoided extra operations.
# MAGIC
# MAGIC - We filled missing data for each column separately. This can be improved by handling all columns in a single operation instead of four separate ones.
# MAGIC
# MAGIC - We recalculated the release year even though it had already been computed earlier. By reusing the existing value, we reduced redundant computation.
# MAGIC
# MAGIC - We also imported the same functions multiple times. There is no need to import col, sum, etc. again if they were already imported earlier. Removing duplicate imports makes the code cleaner and more consistent.

# COMMAND ----------

# 1. Load csv
path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/sales/video_game_sales_2024.csv"
gameSales = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("delimiter", "|")
    .csv(path)
)

# COMMAND ----------

sales_cols = ["na_sales", "jp_sales", "pal_sales", "other_sales"]

# Fill all NULL values in these 4 columns with 0.0 (one operation only)
gameSales = gameSales.na.fill(0.0, subset=sales_cols)


# Remove to date since "release_year" is date
gameSales = gameSales.withColumn(
    "release_year",
    year(col("release_date"))
)

bestJapanPublisher = (
    gameSales
    .filter((col("release_year") >= 2001) & (col("release_year") <= 2010))
    .groupBy("publisher")
    .agg(_sum("jp_sales").alias("japan_sales"))
    .orderBy(col("japan_sales").desc())
)

top_publisher = bestJapanPublisher.first()

# COMMAND ----------

# Use the existing "release_year" column instead of recomputing it.
# No need to import col, sum, etc. again since they were imported earlier.
# Add a global_sales column as the sum of all regional sales
gameSales = gameSales.withColumn(
    "global_sales",
    col("na_sales") + col("jp_sales") + col("pal_sales") + col("other_sales")
)

TOP = top_publisher["publisher"]

bestJapanPublisherSales = (
    gameSales
    .filter(
        (col("publisher") == TOP) &
        (col("release_year") >= 2001) &
        (col("release_year") <= 2010)
    )
    .groupBy("release_year")
    .agg(
        round(_sum("jp_sales"),2).alias("japan_total"),
        round(_sum("global_sales"),2).alias("global_total"),
        round(_sum(
            (col("global_sales") * (col("console") == "PS2").cast("int"))
        ),2).alias("ps2_total")
    )
    .orderBy("release_year")
)

# COMMAND ----------

print(f"The publisher with the highest total video game sales in Japan is: '{bestJapanPublisher}'")
print("Sales data for this publisher:")
bestJapanPublisherSales.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 2
# MAGIC
# MAGIC In the first version, the distance was calculated with a python UDF, which forced Spark to process rows one by one in python and broke many of Spark’s optimizations. We also rounded the distance for every row, even though we only needed the rounded minimum per municipality.
# MAGIC
# MAGIC In the new version we replace the python UDF with Spark’s own math functions and only round the final aggregated value. Also we read only the necessary columns from the parquet file. This reduces overhead, avoids python bottlenecks, and makes the computation faster and more efficient while keeping the same results.

# COMMAND ----------

kampusareenaBuildingId: str = "101060573F"

path = "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/buildings"

# Read only needed columns since parquet is columnar
buildingsDF = (
    spark.read.parquet(path)
    .select(
        "building_id",
        "municipality",
        "street",
        "postal_code",
        "latitude_wgs84",
        "longitude_wgs84",
    )
)

# COMMAND ----------

# one small action
kampus_row = (
    buildingsDF
        .filter(F.col("building_id") == kampusareenaBuildingId)
        .select("latitude_wgs84", "longitude_wgs84")
        .first()
)

kampus_lat = kampus_row["latitude_wgs84"]
kampus_lon = kampus_row["longitude_wgs84"]

R = 6378.1

# compute distance witout UDF, using Spark built-ins
lat1 = F.radians(F.col("latitude_wgs84"))
lon1 = F.radians(F.col("longitude_wgs84"))
lat2 = F.radians(F.lit(kampus_lat))
lon2 = F.radians(F.lit(kampus_lon))

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    F.sin(dlat / 2) ** 2
    + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2
)
distance_km = 2 * F.lit(R) * F.atan2(F.sqrt(a), F.sqrt(1 - a))
buildingsWithDistDF = buildingsDF.withColumn("distance_km", distance_km)


# COMMAND ----------

municipalityDF = (
    buildingsWithDistDF
        .groupBy("municipality")
        .agg(
            F.countDistinct("postal_code").alias("areas"),
            F.countDistinct("street").alias("streets"),
            F.countDistinct("building_id").alias("buildings"),
            F.min("distance_km").alias("min_distance_raw"),
        )
        .withColumn(
            "buildings_per_area",
            F.round(F.col("buildings") / F.col("areas"), 1)
        )
        # round only the final aggregated min_distance
        .withColumn(
            "min_distance",
            F.round(F.col("min_distance_raw"), 1)
        )
        .select(
            "municipality",
            "areas",
            "streets",
            "buildings",
            "buildings_per_area",
            "min_distance",
        )
        .orderBy(F.col("buildings_per_area").desc())
        .limit(10)
)


# COMMAND ----------

print("The 10 municipalities with the highest buildings per area (postal code) ratio:")
municipalityDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 3
# MAGIC
# MAGIC In this part, we forced Spark to:
# MAGIC
# MAGIC - run Python code row by row
# MAGIC - serialize each row
# MAGIC - break DAG optimizations
# MAGIC - lose vectorization
# MAGIC
# MAGIC Because of this, the UDF becomes 10–20× slower than using Spark SQL expressions. Spark cannot optimize or understand Python math functions such as:
# MAGIC
# MAGIC - math.sin
# MAGIC - math.cos
# MAGIC - math.atan2
# MAGIC - math.sqrt
# MAGIC
# MAGIC Spark only understands its own built-in functions, not Python’s math library.
# MAGIC
# MAGIC Another problem was using .first() multiple times, which forces Spark to recompute the entire plan each time. Reducing it to a single .first() call makes the code much faster and more efficient.
# MAGIC
# MAGIC We also repeated code unnecessarily, introduced extra variables, and used redundant references like F.col and F.lit everywhere.
# MAGIC
# MAGIC F.col was used everywhere, and we cleaned this up to make the code simpler.
# MAGIC Another issue was that both Spark and Python have a function called round(). Because I imported Spark’s round, it overrode Python’s built-in round(). As a result, Spark’s round() was accidentally called on a Python float, which caused an error. Using builtins.round solves this problem.
# MAGIC
# MAGIC In general, it is better to use Spark’s round() for DataFrame columns because it is optimized, distributed, and can be pushed down into Spark’s execution plan, unlike Python’s round(), which forces row-by-row processing.

# COMMAND ----------

def haversine_spark(lat1_col, lon1_col, lat2_val, lon2_val):
    lat2_col = lit(lat2_val)
    lon2_col = lit(lon2_val)
    
    return 6378.1 * acos(
        cos(radians(lat1_col)) *
        cos(radians(lat2_col)) *
        cos(radians(lon2_col) - radians(lon1_col)) +
        sin(radians(lat1_col)) * sin(radians(lat2_col))
    )

def closest_to_average(df):

    # 1) Remove duplicate buildings
    df_unique = df.dropDuplicates(["building_id"])

    # 2) Compute the average coordinates (ONE ACTION)
    avg_vals = df_unique.agg(
        avg("latitude_wgs84").alias("avg_lat"),
        avg("longitude_wgs84").alias("avg_lon")
    ).first()   #(avg_lat, avg_lon) we need these two numbers in Python because we later use them inside lit(avg_lat).

    avg_lat, avg_lon = float(avg_vals.avg_lat), float(avg_vals.avg_lon)

    # 3) Compute distances (no UDF = fast)
    df_dist = df_unique.withColumn(
        "dist_to_avg",
        haversine_spark(
            col("latitude_wgs84"),
            col("longitude_wgs84"),
            lit(avg_lat),
            lit(avg_lon),
        )
    )

    # 4) Closest building (ONE ACTION)
    #After computing the distances, you want the single closest row.
    # Again, it is a legitimate action.
    row = df_dist.orderBy("dist_to_avg").first()  

    return {
        "avg_lat": avg_lat,
        "avg_lon": avg_lon,
        "closest_street": row["street"],
        "closest_house_number": row["house_number"],
        "distance_km": float(row["dist_to_avg"])
    }


# COMMAND ----------


# postal code 33720
tampereDF = buildingsDF.filter(col("municipality") == "Tampere")
hervantaDF = buildingsDF.filter(col("postal_code") == 33720)

result_tampere = closest_to_average(tampereDF)
result_hervanta = closest_to_average(hervantaDF)


tampereAddress = f"{result_tampere['closest_street']} {result_tampere['closest_house_number']}"
hervantaAddress = f"{result_hervanta['closest_street']} {result_hervanta['closest_house_number']}"

# Use the built-in round for floats
tampereDistance = __builtins__.round(result_tampere['distance_km'], 3)
hervantaDistance = __builtins__.round(result_hervanta['distance_km'], 3)

# COMMAND ----------

print(f"The address closest to the average location in Tampere: '{tampereAddress}' at ({tampereDistance} km)")
print(f"The address closest to the average location in Hervanta: '{hervantaAddress}' at ({hervantaDistance} km)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 4
# MAGIC
# MAGIC For Advanced Task 4 we try to keep things simple:
# MAGIC - First we only pick the columns we really need from players and events, so less I/O and memory.
# MAGIC - Then we use inner join, because we only care about events that actually belong to players, no need for extra outer rows.
# MAGIC - We also filter early, keeping just scoring events (Shot, Free Kick) with Goal or Own goal tags, and skip goalkeeper ones. This way the data is smaller before aggregation.
# MAGIC - We cached the joined and goal_events DataFrames, since we use them again for Spain and Italy queries, so no recomputation.
# MAGIC - Finally we group and aggregate only the needed cols, and with orderBy(...).limit(7) we get top 7 scorers fast and efficient.

# COMMAND ----------

# Select only needed columns (avoid wide scans) ---
players_small = playerDF.select("playerId", "firstName", "lastName")
events_small  = eventsDF.select("eventPlayerId", "eventTeam", "event", "tags", "competition", "season")

# Join players with events (inner: we only need events that have players) ---
# inner join reduces rows compared to outer; cache because we reuse the result
joined = players_small.join(
    events_small,
    players_small.playerId == events_small.eventPlayerId,
    how="inner"
).cache()

# Filter early to keep only scorer events
goal_events = joined.filter(
    (F.col("event").isin("Shot", "Free Kick")) &
    (F.array_contains(F.col("tags"), "Goal") | F.array_contains(F.col("tags"), "Own goal")) &
    (~F.array_contains(F.col("tags"), "Goalkeeper"))
).select(
    F.concat_ws(" ", F.col("firstName"), F.col("lastName")).alias("player"),
    F.col("eventTeam").alias("team"),
    F.col("competition"),
    F.col("season")
).cache()   # cached because used twice (Spain + Italy)

# Top 7 goalscorers in Spanish La Liga 2017-2018 
goalscorersSpainDF = (
    goal_events
    .filter((F.col("competition") == "Spanish La Liga") & (F.col("season") == "2017-2018"))
    .groupBy("player", "team")
    .agg(F.count("*").alias("goals"))
    .orderBy(F.desc("goals"))
    .limit(7)
)

# Top 7 goalscorers in Italian Serie A 2017-2018 ---
goalscorersItalyDF = (
    goal_events
    .filter((F.col("competition") == "Italian Serie A") & (F.col("season") == "2017-2018"))
    .groupBy("player", "team")
    .agg(F.count("*").alias("goals"))
    .orderBy(F.desc("goals"))
    .limit(7)
)

# Release cached intermediate dataframes if not needed further ---
joined.unpersist()
goal_events.unpersist()

print("The top 7 goalscorers in Spanish La Liga in season 2017-18:")
goalscorersSpainDF.show(truncate=False)

print("The top 7 goalscorers in Italian Serie A in season 2017-18:")
goalscorersItalyDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 5
# MAGIC For Advanced Task 1 I try to make the process efficient:
# MAGIC
# MAGIC - We only choose the columns we really need from players and events, so less I/O and memory used.
# MAGIC - Then we filter early by season and event types, so later joins and aggregations work on smaller dataset.
# MAGIC - We use inner join between Finnish players and filtered events, because we only care about events linked to Finnish players, no extra rows.
# MAGIC - After that we aggregate by (playerId, matchId) and keep only ev_count ≥ 3, then count these rows per player. This way we avoid expensive countDistinct and too many shuffles.
# MAGIC - Finally we do a left join to also include Finnish players with zero appearances, and order the result by matches.

# COMMAND ----------

# Read/select only necessary player columns and restrict to Finnish players
players_small = playerDF.select("playerId", "firstName", "lastName", "birthArea")
finn_players = players_small.filter(col("birthArea") == "Finland") \
                            .select("playerId", "firstName", "lastName")

# Select only needed event columns and filter early (season + event types)
events_small = eventsDF.select("eventPlayerId", "event", "matchId", "season") \
    .filter(col("season") == "2017-2018") \
    .filter(col("event").isin("Shot", "Free Kick", "Pass", "Save attempt")) \
    .filter(col("eventPlayerId") != 0)  # ignore events not attributed to a player

# Join Finnish players with their (filtered) events — inner join keeps dataset small
finn_events = finn_players.join(
    events_small,
    finn_players.playerId == events_small.eventPlayerId,
    how="inner"
).select(
    finn_players.playerId.alias("playerId"),
    "matchId"
)

# Count events per player per match (one row per player-match)
events_per_match = finn_events.groupBy("playerId", "matchId").agg(count("*").alias("ev_count"))

# A match appearance = player has at least 3 such events in the match
appearances = events_per_match.filter(col("ev_count") >= 3)

# For each player count how many matches they appeared in
matches_per_player = appearances.groupBy("playerId").agg(count("*").alias("matches"))

# Left-join to include Finnish players with zero appearances and produce final names
finnishPlayersDF = finn_players.select("playerId", "firstName", "lastName") \
    .join(matches_per_player, on="playerId", how="left") \
    .na.fill({"matches": 0}) \
    .select(
        concat_ws(" ", col("firstName"), col("lastName")).alias("player"),
        col("matches").cast("int")
    ) \
    .orderBy(desc("matches"))

# Show result (or return/use finnishPlayersDF)
finnishPlayersDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 6
# MAGIC For Advanced Task we tried to optimize the pipeline:
# MAGIC - First we only keep needed cols from players (playerId, name, birthArea) so less data size.
# MAGIC - Then we filter events early by season and event types, so later joins and groups are smaller.
# MAGIC - We join events with players, using broadcast since players table is small, so shuffle avoided.
# MAGIC - After that we count events per player per match+competition, and keep only appearances with ≥3 events.
# MAGIC - We group again to count matches per player per competition, and cache because we use it twice.
# MAGIC - With a Window we rank competitions per player by matches (desc, tie breaker alphabetical).
# MAGIC - Then we take top1 and top2 competitions, join them, so only players with at least two comps remain.
# MAGIC - Finally we attach birthArea and filter to keep players with ≥10 matches in both competitions.
# MAGIC

# COMMAND ----------

#Add alias as helper function
spark_count = count
# Project only needed columns from players (reduce data size)
players_small = (
    playerDF
    .select(
        col("playerId"),
        concat_ws(" ", col("firstName"), col("lastName")).alias("player"),
        col("birthArea")
    )
    .distinct()
    # Cache() if players_small reused many times and is not tiny
)

# 2) Filter events early: season + event types; select only needed columns
events_needed = ["Shot", "Pass", "Free Kick", "Save attempt"]
events_filtered = (
    eventsDF
    .filter((col("season") == "2017-2018") & col("event").isin(events_needed))
    .select(
        col("eventPlayerId").alias("playerId"),
        col("matchId"),
        col("competition")
    )
)

# Join filtered events with players (broadcast players if small) to attach player name
#    Using broadcast avoids a shuffle if players_small is small
joinedDF = events_filtered.join(
    broadcast(players_small.select("playerId", "player")),
    on="playerId",
    how="inner"
)

# Count events per player per match per competition and keep only appearances (>=3 events)
eventsPerMatch = (
    joinedDF
    .groupBy("player", "matchId", "competition")
    .agg(spark_count("*").alias("event_count"))
    .filter(col("event_count") >= 3)
    # After this point, each row == one match appearance (player, match, competition)
)

# Count matches (appearances) per player per competition
appearancesPerCompetition = (
    eventsPerMatch
    .groupBy("player", "competition")
    .agg(spark_count("*").alias("matches"))
)

# Cache because we will scan this frame twice for top1 and top2
appearancesPerCompetition = appearancesPerCompetition.cache()

# Use Window to rank competitions per player by matches desc, competition asc (tie-breaker)
w = Window.partitionBy("player").orderBy(col("matches").desc(), col("competition").asc())

ranked = appearancesPerCompetition.withColumn("rn", row_number().over(w))

# Extract top1 and top2 competitions per player
comp1 = ranked.filter(col("rn") == 1).select(
    col("player"), col("competition").alias("competition1"), col("matches").alias("matches1")
)

comp2 = ranked.filter(col("rn") == 2).select(
    col("player"), col("competition").alias("competition2"), col("matches").alias("matches2")
)

# Join comp1 and comp2 to keep players who have two distinct competitions
#    This is an inner join -> ensures player has at least two competitions
two_comps = comp1.join(comp2, on="player", how="inner")

# Attach birthArea (left join — in case birthArea missing)
player_birth = players_small.select("player", "birthArea").distinct()

appearanceDF: DataFrame = (
    two_comps
    .join(player_birth, on="player", how="left")
    # 10) Final filter: both competitions must have at least 10 matches
    .filter((col("matches1") >= 10) & (col("matches2") >= 10))
    .select("player", "birthArea", "competition1", "matches1", "competition2", "matches2")
)

# Show final result (single action)
print("Players who played in at least 10 matches in two separate competitions:")
appearanceDF.show(truncate=False)

# Optionally unpersist cached DF to free memory
appearancesPerCompetition.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized and correct solution for Task 7
# MAGIC - Early filtering: Only necessary columns selected and data filtered by season/competition at start, reducing size and I/O.
# MAGIC MatchInfo: A single unique matchInfoDF created and cached, preventing redundant computation in later joins.
# MAGIC - Joins & shuffles: Three separate joins for goals replaced with one Left Outer Join and groupBy using spark_max, which reduced shuffle operations.
# MAGIC - Caching: teamWinsDF cached since needed in two actions (count and bestTeams).
# MAGIC - Actions: collect() and other driver‑side operations avoided; null values handled directly with fillna(0) in Spark.
# MAGIC - Max aggregation: spark_max(struct(...)) used to find top team per competition efficiently, avoiding costly orderBy or Window functions.

# COMMAND ----------

# from pyspark.sql.functions import (
#     col, array_contains, count as spark_count, when,
#     max as spark_max, struct, lit
# )

# 1. Early Projection and Filtering on initial DataFrame
# Assuming eventsDF is already loaded and contains all columns.
# Select only necessary columns and filter early for the specific season and competition.
eventsDF = eventsDF.select(
    "matchId", "season", "competition",
    "homeTeam", "awayTeam", "event",
    "eventTeam", "tags"
).filter(
    (col("season") == "2017-2018") &
    col("competition").isNotNull()
)

# 2. Extract Match Info ONCE (Minimal Duplicates)
# Extract unique match information and cache it, as it will be used for multiple joins.
matchInfoDF = eventsDF.select(
    "matchId", "homeTeam", "awayTeam", "competition"
).dropDuplicates(["matchId"]).cache()

# 3. Simplify Goal Calculation (Single Select/Join approach)

# A. Extract regular goals (Tags include 'Goal')
regularGoalsDF = eventsDF.filter(
    array_contains(col("tags"), "Goal") &
    (col("event") != "Save attempt")
).select(
    "matchId", col("eventTeam").alias("scoringTeam"), lit(1).alias("goalCount")
)

# B. Extract own goals and determine the scoring team (the opponent) in a single Join
ownGoalsDF = eventsDF.filter(
    array_contains(col("tags"), "Own goal") &
    (col("event") != "Save attempt")
).alias("og").join(
    matchInfoDF.select("matchId", "homeTeam", "awayTeam").alias("m"),
    col("og.matchId") == col("m.matchId")
).withColumn(
    "scoringTeam",
    # The scoring team is the opposite team
    when(col("og.eventTeam") == col("m.homeTeam"), col("m.awayTeam"))
    .otherwise(col("m.homeTeam"))
).select(col("og.matchId").alias("matchId"), "scoringTeam", lit(1).alias("goalCount"))

# 4. Combine goals and aggregate goals per (match, team)
# Use unionByName for safer column alignment.
allGoalsDF = regularGoalsDF.unionByName(ownGoalsDF)

# Aggregate goals per (match, team)
goalsPerTeamDF = (
    allGoalsDF.groupBy("matchId", "scoringTeam")
              .agg(spark_count("*").alias("goals"))
)

# 5. Single Join to Match Info and Calculate Scores
# Perform a single left join to combine match details with goal counts.
matchesWithGoalsDF = matchInfoDF.alias("m").join(
    goalsPerTeamDF.alias("g"),
    ["matchId"],
    "leftouter"
).withColumn(
    "homeGoals_raw",
    # now we use the columns from the unified DataFrame
    when(col("homeTeam") == col("g.scoringTeam"), col("g.goals")).otherwise(lit(0))
).withColumn(
    "awayGoals_raw",
    when(col("awayTeam") == col("g.scoringTeam"), col("g.goals")).otherwise(lit(0))
)

# Aggregate the raw goal counts to get the final score for each match
# The grouping columns (matchId, competition, homeTeam, awayTeam) are now unambiguous.
matchesWithGoalsDF = matchesWithGoalsDF.groupBy(
    "matchId", "competition", "homeTeam", "awayTeam"
).agg(
    spark_max("homeGoals_raw").alias("homeGoals"),
    spark_max("awayGoals_raw").alias("awayGoals")
).fillna(0, ["homeGoals", "awayGoals"])

# 6. Determine match winner
matchesWithWinnerDF = matchesWithGoalsDF.withColumn(
    "winner",
    when(col("homeGoals") > col("awayGoals"), col("homeTeam"))
    .when(col("awayGoals") > col("homeGoals"), col("awayTeam"))
)

# 7. Count total wins per team (Optimized)
# Cache the wins calculation as it is used for two subsequent actions (count and final output).
teamWinsDF = (
    matchesWithWinnerDF.filter(col("winner").isNotNull())
                        .groupBy(col("winner").alias("team"))
                        .agg(spark_count("*").alias("wins"))
                        .cache()
)

# Count teams with ≥ 20 wins (First Action)
twentyWinsCount = teamWinsDF.filter(col("wins") >= 20).count()

# 8. Best team per competition
# Calculate wins per competition and team
teamWinsCompDF = (
    matchesWithWinnerDF.filter(col("winner").isNotNull())
                        .groupBy("competition", col("winner").alias("team"))
                        .agg(spark_count("*").alias("wins"))
)

# Find the best team per competition using struct and max aggregation
bestTeamsDF = (
    teamWinsCompDF.groupBy("competition")
                  .agg(spark_max(struct(col("wins"), col("team"))).alias("max_struct"))
                  .select(
                      col("competition"),
                      col("max_struct.team").alias("team"),
                      col("max_struct.wins").alias("wins")
                  )
                  .orderBy(col("wins").desc())
)

print(f"Number of teams having at least 20 wins: {twentyWinsCount}")
print("The teams with the most wins in each competition:")
bestTeamsDF.show(truncate=False)
