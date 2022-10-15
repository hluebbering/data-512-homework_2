# Homework 2. Considering Bias in Data

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

<img height="32" width="32" src="https://user-images.githubusercontent.com/61786322/195970564-6ff7fa35-43bb-4ff6-b415-e537e099074a.svg"/>

https://pagespeed-insights.herokuapp.com?url=https://ankurparihar.github.io



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

```python
politician_articles = pd.read_csv('data/politicians_by_country_SEPT.2022.csv')
politician_articles.head()
```






------------------------------------



## Research Implications


> Reflect on what you have learned, what you found, what (if anything) surprised you about your findings, and/or what theories you have about why any biases might exist (if you find they exist). 


1. What biases did you expect to find in the data (before you started working with it), and why?
2. What (potential) sources of bias did you discover in the course of your data processing and analysis?
3. What might your results suggest about (English) Wikipedia as a data source?
4. What might your results suggest about the internet and global society in general?
5. Can you think of a realistic data science research situation where using these data (to train a model, perform a hypothesis-driven research, or make business decisions) might create biased or misleading results, due to the inherent gaps and limitations of the data?
6. Can you think of a realistic data science research situation where using these data (to train a model, perform a hypothesis-driven research, or make business decisions) might still be appropriate and useful, despite its inherent limitations and biases?
How might a researcher supplement or transform this dataset to potentially correct for the limitations/biases you observed?




