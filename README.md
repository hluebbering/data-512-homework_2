
# Homework 2. Considering Bias in Data

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Wikipedia](https://img.shields.io/badge/Wikipedia-%23000000.svg?style=for-the-badge&logo=wikipedia&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)


<br>



> ![CSS3](https://img.shields.io/badge/Project-Goal:-%23008080.svg?style=plastic&logo=star&logoColor=white)
The following explores the concept of data bias using Wikipedia articles considering political figures from different countries. For this project, we combine a dataset of Wikipedia articles with a dataset of country populations. We then use `ORES`, a machine learning service, to estimate the quality of each article.

We perform an analysis of how the coverage of politicians on Wikipedia and the quality of articles about politicians varies among countries. The analysis consists of a series of tables that show:


- ✅ Country coverage of politicians on Wikipedia compared to their population</br>
- ✅ Country proportion of high quality articles about politicians</br>
- ✅ Ranking of regions by articles-per-person and proportion of high quality articles</br>

Lastly, the reflection focuses on how the project analysis findings and the process to reach those findings help understand the causes and consequences of biased data in large, complex data science projects.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML
from IPython.core.display import HTML

def css_styling(): # Styling notebook
    styles = open("./data/custom.css", "r").read()
    return HTML(styles)
css_styling()
```






----------------------------------------------------



1. <code class = "mycode">politicians_by_country.SEPT.2022.csv</code>: a list of article pages about politicians from different countries crawled from the Wikipedia [Category: Politicians by nationality](https://en.wikipedia.org/wiki/Category:Politicians_by_nationality)



```python
politician_articles = pd.read_csv('data/politicians_by_country_SEPT.2022.csv')
politician_articles.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>url</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shahjahan Noori</td>
      <td>https://en.wikipedia.org/wiki/Shahjahan_Noori</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abdul Ghafar Lakanwal</td>
      <td>https://en.wikipedia.org/wiki/Abdul_Ghafar_Lak...</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Majah Ha Adrif</td>
      <td>https://en.wikipedia.org/wiki/Majah_Ha_Adrif</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Haroon al-Afghani</td>
      <td>https://en.wikipedia.org/wiki/Haroon_al-Afghani</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tayyab Agha</td>
      <td>https://en.wikipedia.org/wiki/Tayyab_Agha</td>
      <td>Afghanistan</td>
    </tr>
  </tbody>
</table>
</div>



&#10148; *NOTE.* Data crawling Wikipedia to identify page subsets might result in misleading and/or duplicate category labels. Document any data inconsistencies and how to handle them.


2. <code class = "mycode">population_by_country_2022.csv</code>: country populations data drawn from the [world population data sheet](https://www.prb.org/international/indicator/population/table) published by the Population Reference Bureau.


```python
country_populations = pd.read_csv('data/population_by_country_2022.csv')
country_populations.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Geography</th>
      <th>Population (millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WORLD</td>
      <td>7963.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AFRICA</td>
      <td>1419.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NORTHERN AFRICA</td>
      <td>251.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Algeria</td>
      <td>44.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Egypt</td>
      <td>103.5</td>
    </tr>
  </tbody>
</table>
</div>



----------------------------------------------------


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




