# Homework 2. Considering Bias in Data


> **Project Goal:** The following explores the concept of data bias using Wikipedia articles considering political figures from different countries. For this project, we combine a dataset of Wikipedia articles with a dataset of country populations. We then use `ORES`, a machine learning service, to estimate the quality of each article.

We perform an analysis of how the coverage of politicians on Wikipedia and the quality of articles about politicians varies among countries. The analysis consists of a series of tables that show:

- countries with the greatest and least coverage of politicians on Wikipedia compared to their population
- countries with the highest and lowest proportion of high quality articles about politicians
- a ranking of geographic regions by articles-per-person and proportion of high quality articles


Lastly, the reflection focuses on how the project analysis findings and the process to reach those findings help understand the causes and consequences of biased data in large, complex data science projects.

----------------------------------------------------


## Step 1: Getting the Article and Population Data


We need data that lists `Wikipedia articles of politicians` and data for `country populations`.


1. `politicians_by_country.SEPT.2022.csv`: a list of article pages about politicians from different countries crawled from the Wikipedia [Category: Politicians by nationality](https://en.wikipedia.org/wiki/Category:Politicians_by_nationality)
