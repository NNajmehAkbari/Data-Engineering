# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2025 Tampere University<br>
# MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
# MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
# MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

# COMMAND ----------

# MAGIC %md
# MAGIC # COMP.CS.320 Data-Intensive Programming, Exercise 1
# MAGIC
# MAGIC This exercise is mostly introduction to the Azure Databricks notebook system.
# MAGIC
# MAGIC There are some basic programming tasks that can be done in either Scala or Python. The Spark related tasks start next week in exercise 2.
# MAGIC
# MAGIC This is the **Python** version, switch to the Scala version if you want to do the tasks in Scala.
# MAGIC
# MAGIC Each task has its own cell(s) for the code. Add your solutions to the cells. You are free to add more cells if you feel it is necessary.<br>
# MAGIC There are cells with test code and example output following most of the tasks that involve producing code.
# MAGIC
# MAGIC At the end of the notebook, there is a question regarding the use of AI or other collaboration when working the tasks.<br>
# MAGIC Please remember to answer the AI question. And finally, don't forget to submit your solutions to Moodle: [Weekly Exercise #1](https://moodle.tuni.fi/mod/assign/view.php?id=3503816)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Read tutorial
# MAGIC
# MAGIC Read the "[Basics of using Databricks notebooks](https://adb-7895492183558578.18.azuredatabricks.net/editor/notebooks/743402606902162)" tutorial notebook.<br>
# MAGIC It is in a read-only folder, so to make edits clone the tutorial notebook to your own workspace and run at least the first couple of code examples.
# MAGIC
# MAGIC - File -> Clone...
# MAGIC - Modify the notebook name if you want
# MAGIC - Select your workspace folder (Workspace/Users/YOUR_TUNI_EMAIL) using Browse
# MAGIC - Clone
# MAGIC - Now a clone of the notebook is in your folder and you can edit the cells. "tuni-cs320-f2025-student" cluster should be available for you to run and test the notebook.
# MAGIC
# MAGIC To get a point from this task, replace "???" with "Task 1 is done" (or something similar) in the following cell (after you have read the tutorial).

# COMMAND ----------

# MAGIC %md
# MAGIC Task 1 is done
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Basic function
# MAGIC
# MAGIC Part 1:
# MAGIC
# MAGIC - Write a simple function `mySum` that takes two integers as parameters and returns their sum.
# MAGIC
# MAGIC Part 2:
# MAGIC
# MAGIC - Write a function `myTripleAvg` that takes three integers as parameters and returns their average rounded to the nearest integer.

# COMMAND ----------

def mySum(x: int, y: int):
    return x+y
def myTripleAvg(x: int, y: int, z:int):
    return round((x+y+z)/3)
mySum(1,2)

# COMMAND ----------

# Code to test your functions:
print(f"- mySum(20, 21) = {mySum(20, 21)}")
print(f"- mySum(-2, 17) = {mySum(-2, 17)}")
print(f"- myTripleAvg(20, 21, 23) = {myTripleAvg(20, 21, 23)}")
print(f"- myTripleAvg(20, 21, 24) = {myTripleAvg(20, 21, 24)}")
print(f"- myTripleAvg(12, -9, -3) = {myTripleAvg(12, -9, -3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC - mySum(20, 21) = 41
# MAGIC - mySum(-2, 17) = 15
# MAGIC - myTripleAvg(20, 21, 23) = 21
# MAGIC - myTripleAvg(20, 21, 24) = 22
# MAGIC - myTripleAvg(12, -9, -3) = 0
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Fibonacci numbers
# MAGIC
# MAGIC The Fibonacci numbers, `F_n`, are defined such that each number is the sum of the two preceding numbers. The first two Fibonacci numbers are:
# MAGIC
# MAGIC $$F_0 = 0 \qquad F_1 = 1$$
# MAGIC
# MAGIC In the following cell, write a **recursive** function, `fibonacci`, that takes in the index and returns the Fibonacci number. (no need for any optimized solution here)

# COMMAND ----------

def fibonacci(n: int):
    if n == 0:
        return 0
    if n == 1:  
        return 1
    return fibonacci(n-1) + fibonacci(n-2)
    


# COMMAND ----------

print(f"- fibonacci(1) = {fibonacci(1)}")
print(f"- fibonacci(5) = {fibonacci(5)}")
print(f"- fibonacci(9) = {fibonacci(9)}")
print(f"- fibonacci(12) = {fibonacci(12)}")
print(f"- fibonacci(17) = {fibonacci(17)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC - fibonacci(1) = 1
# MAGIC - fibonacci(5) = 5
# MAGIC - fibonacci(9) = 34
# MAGIC - fibonacci(12) = 144
# MAGIC - fibonacci(17) = 1597
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Map and reduce
# MAGIC
# MAGIC - `map` function can be used to transform the elements of a list.
# MAGIC - `reduce` function can be used to combine the elements of a list into a single value.
# MAGIC
# MAGIC Part 1:
# MAGIC
# MAGIC - Using the `myList`as a starting point, use function `map` to calculate the cube of each element
# MAGIC - and then use the `reduce` function to calculate the sum of the cubes.
# MAGIC
# MAGIC Part 2:
# MAGIC
# MAGIC - Using functions `map` and `reduce`, find the smallest value for `f(x)=x^2-34*x+325` when the input values `x` are the values from `myList`.

# COMMAND ----------

from functools import reduce

myList: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 21, 23, 29]

cubeSum: int = reduce(
    lambda x, y: x + y,
    map(lambda x: x ** 3, myList)
)
smallestFuncValue: int = reduce(
    lambda x, y: min(x, y),
    map(lambda x: x ** 2 - 34 * x + 325, myList)
)

# COMMAND ----------

print(f"Sum of cubes:                        {cubeSum}")
print(f"Smallest value of f(x)=x^2-34*x+325:    {smallestFuncValue}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC Sum of cubes:                        61620
# MAGIC Smallest value of f(x)=x^2-34*x+325:    36
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Grouping and aggregating
# MAGIC
# MAGIC You are given a list of tuples representing sensor readings (`temperatureReadings`). Each tuple contains a sensor name and a temperature reading.<br>
# MAGIC You are also given a function `average` that returns the average value of the given list.
# MAGIC
# MAGIC Part 1:
# MAGIC
# MAGIC - Using the given readings and the `average` function, calculate the average temperature reading for each sensor and assign them as a dictionary to `averageTemperatures`.<br>
# MAGIC Any working solution is accepted, but you are encouraged to try to solve this using `groupby` function from itertools.
# MAGIC
# MAGIC Part 2:
# MAGIC
# MAGIC - Find the value for the highest average temperature.

# COMMAND ----------

from functools import reduce
from itertools import groupby  # itertools.groupby requires the list to be sorted

temperatureReadings: list[tuple[str, float]] = [
    ("sensorA", 19.4), ("sensorB", 19.0), ("sensorA", 20.1), ("sensorC", 18.6),
    ("sensorC", 18.7), ("sensorB", 20.5), ("sensorC", 18.3), ("sensorA", 18.3),
    ("sensorA", 21.0), ("sensorB", 21.1), ("sensorC", 19.2), ("sensorB", 19.8)
]

def average(values: list[float]) -> float:
    value_sum, value_count = reduce(
        lambda pair, elem: (pair[0] + elem, pair[1] + 1),
        values,
        (0.0, 0)
    )
    return value_sum / value_count

# A map for the average temperatures for each sensor
averageTemperaturesSort = sorted(temperatureReadings, key=lambda x: x[0])
averageTemperatures = {
    name: average(list(map(lambda x: x[1], group)))
    for name, group in groupby(averageTemperaturesSort, key=lambda x: x[0])
}

highestAverageTemperature = max(averageTemperatures.values())
#averageTemperatures


# COMMAND ----------

print(f"type(averageTemperatures): {type(averageTemperatures)} - type(averageTemperatures['sensorA']): {type(averageTemperatures['sensorA'])}")
print()
print("Average temperatures per sensor:")
for name, avg in averageTemperatures.items():
    print(f"{name}: {avg}")
print(f"The highest average temperature: {highestAverageTemperature}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output:
# MAGIC
# MAGIC ```text
# MAGIC type(averageTemperatures): <class 'dict'> - type(averageTemperatures['sensorA']): <class 'float'>
# MAGIC
# MAGIC Average temperatures per sensor:
# MAGIC sensorA: 19.7
# MAGIC sensorB: 20.1
# MAGIC sensorC: 18.7
# MAGIC The highest average temperature: 20.1
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Code explanation
# MAGIC
# MAGIC Look at the `average` function that was given in task 5.
# MAGIC
# MAGIC If you are not used to functional programming, it might look quite strange and somewhat different than the way you would "traditionally" write a function that calculates the average from a list of numbers (when doing so without using any built-in functionalities of the programming language).
# MAGIC
# MAGIC - Explain in your own words how the `average` function works.
# MAGIC - Are there any benefits to this implementation compared to a straightforward implementation that separately calculates the sum and the count of the elements?
# MAGIC - What happens with this implementation if the input is an empty list? Given your solution to task 5, can this cause issues?

# COMMAND ----------

# MAGIC %md
# MAGIC It groups items based on first elements and gives average of second items as value, for example it goups by "sensorA" gives average of 19.4 + 20.1 + 18.3 + 21.0 = 78.8/4 = 19.7

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Approximation for seventh root
# MAGIC
# MAGIC Write a function, `seventhRoot`, that returns an approximate value for the seventh root of the input.<br>
# MAGIC Use the Newton's method, [https://en.wikipedia.org/wiki/Newton's_method](https://en.wikipedia.org/wiki/Newton%27s_method), with the initial guess of 1. For the seventh root this Newton's method translates to:
# MAGIC
# MAGIC $$y_0 = 1$$
# MAGIC $$y_{n+1} = \frac{1}{7}\bigg(6y_n + \frac{x}{y_n^6}\bigg) $$
# MAGIC
# MAGIC where `x` is the input value and `y_n` is the guess for the cube root after `n` iterations. For large enough `k`, $$y_k \approx \sqrt[7]{x} $$
# MAGIC
# MAGIC Example steps when `x=128`:
# MAGIC
# MAGIC $$y_0 = 1$$
# MAGIC $$y_1 = \frac{1}{7}\big(6*1 + \frac{128}{1^6}\big) \approx 19.143$$
# MAGIC
# MAGIC $$y_2 \approx \frac{1}{7}\big(6*19.143 + \frac{128}{19.143^6}\big) \approx 16.408$$
# MAGIC
# MAGIC $$y_3 \approx \frac{1}{7}\big(6*16.408 + \frac{128}{16.408^6}\big) \approx 14.064$$
# MAGIC
# MAGIC $$...$$
# MAGIC
# MAGIC You will have to decide yourself on what is the condition for stopping the iterations. (you can add parameters to the function if you think it is necessary)
# MAGIC
# MAGIC Note, if your code is running for hundreds or thousands of iterations, you are either doing something wrong or trying to calculate too precise values.<br>
# MAGIC The example output has been calculated using less than 100 iterations in each case.

# COMMAND ----------

def seventhRoot(x: float) -> float:
    if x==0:
        return 0
    result = 1
    for i in range(100):
        result = 1/7*(6*result + x/(result**6))
        
    return result    

# COMMAND ----------

print(f"Seventh root of 128:       {seventhRoot(128)}")
print(f"Seventh root of 78125:     {seventhRoot(78125)}")
print(f"Seventh root of 10^7:     {seventhRoot(1e7)}")
print(f"Seventh root of 10^(-7):   {seventhRoot(1e-7)}")
print(f"Seventh root of -2187:    {seventhRoot(-2187)}")
print(f"Seventh root of 0:         {seventhRoot(0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output
# MAGIC
# MAGIC (the exact values are not important, but the results should be close enough)
# MAGIC
# MAGIC ```text
# MAGIC Seventh root of 128:       2.0000000000000067
# MAGIC Seventh root of 78125:     5.000000000023987
# MAGIC Seventh root of 10^7:     10.000000000033177
# MAGIC Seventh root of 10^(-7):   0.10000000000010757
# MAGIC Seventh root of -2187:    -3.0000000000751648
# MAGIC Seventh root of 0:         0.0
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - General programming background question
# MAGIC
# MAGIC For the final task of the first exercise, just answer the following questions:
# MAGIC
# MAGIC 1. What king of programming background do you have? (for example, how many courses have you taken or how many years of experience do you have)
# MAGIC 2. How did you find the use of the Databricks environment and the notebooks?
# MAGIC 3. How did you find these programming tasks? (easy, hard, interesting, annoying, ...)
# MAGIC
# MAGIC There are no correct answers to this task.<br>
# MAGIC The answers might give the course staff a better understanding on level of the programming skills the students have.<br>
# MAGIC Short answers are accepted, but you are free to write any constructive comments you can think of.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 1. I know Python and C++
# MAGIC 2. Not very chalanging, I enjoy working with it.
# MAGIC 3. It was intersting and I learned new things, I wasn't familiare with for example functools.
# MAGIC

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
# MAGIC
# MAGIC For correcting some syntax.
# MAGIC I did them alone.
