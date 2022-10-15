
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



----------------------------------------------------


## Step 1: Getting the Article and Population Data



<div class="alert alert-block alert-info">

We get data that lists **Wikipedia articles of politicians** and data for **country populations**.

</div>

1. <code class = "mycode">politicians_by_country.SEPT.2022.csv</code>: a list of article pages about politicians from different countries crawled from the Wikipedia [Category: Politicians by nationality](https://en.wikipedia.org/wiki/Category:Politicians_by_nationality)



```python
politician_articles = pd.read_csv('data/politicians_by_country_SEPT.2022.csv')
politician_articles.head()
```




<div>

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



## Step 2: Getting Article Quality Predictions

<div class="alert alert-block alert-info">

Now we get the `predicted quality scores` for each article in the Wikipedia dataset using `ORES`.
    
</div>

ORES is a machine learning tool that provides estimates of Wikipedia article quality. 
- The article quality estimates are, from best to worst:
    - `FA:` Featured article
    - `GA:` Good article
    - `B:` B-class article
    - `C:` C-class article
    - `Start:` Start-class article
    - `Stub:` Stub-class article


```python
# List of Wikipedia article pages about politicians
politician_names = []
for name in politician_articles['name']:
    politician_names.append(name)
```

### MediaWiki API: Making Page Info Request

ORES requires a specific revision ID of a specific article to be able to make a label prediction. We use the `MediaWiki REST API` [API:Info](https://www.mediawiki.org/wiki/API:Info) request to get a range of metadata on an article, including the most current revision ID of the article page. For each Wikipedia article from our list of article pages about politicians, we make a page info request to get the current page revision id used for ORES scoring. We then save the article revisions dictionary to JSON file.



```python
# Dictionary of Wikipedia article titles (keys) and revision IDs used for ORES scoring
ARTICLE_REVISIONS = {}

# Maintain a log of articles not able to retrieve an ORES score.
article_without_ORES_score = []

for name in politician_names:
    
    # Make a page info request to get the current page revision
    info = request_pageinfo_per_article(name)    
    
    for i in info['query']['pages'].items():
        
        # If unable to get a score for a particular article
        if i[0] == '-1':
            
            article_without_ORES_score.append(name)
            continue
        
        # Match article title to specific revision ID
        else:
            item_values = i[1]
            ARTICLE_REVISIONS[name] = item_values['lastrevid']


# Save article revisions dictionary to JSON file
with open("data/ARTICLE_REVISIONS.json", "w") as outfile:
    json.dump(ARTICLE_REVISIONS, outfile, indent = 4)
```

**`Note.`** Some articles have no score. Below is a log of articles for which we were not able to retrieve an ORES score.


Articles with no retrievable ORES score:

- 'Prince Ofosu Sefah'
- 'Harjit Kaur Talwandi'
- 'Abd al-Razzaq al-Hasani'
- 'Kang Sun-nam'
- 'Abiodun Abimbola Orekoya'
- 'Segun “Aeroland” Adewale'
- 'Roman Konoplev'
- 'Nhlanhla “Lux” Dlamini']



### Scores API: Making an ORES Request


Here, we generate **quality scores for article revisions using ORES**. This example shows how to request a score of a specific revision, where the score provides probabilities for all of the possible article quality levels. The API request will be made using one procedure. The idea is to make this reusable. The procedure is parameterized, but relies on the constants above for the important parameters. The underlying assumption is that this will be used to request data for a set of article revisions. Therefore, the main parameter is article_revid.


For each article page, we make an ORES request using the `page title` and current `revision id`. We saved this information in the `ARTICLE_REVISIONS.json` file. 


```python
# Open ARTICLE_REVISIONS file
with open('data/ARTICLE_REVISIONS.json', 'r') as openfile:
    
    # Read from json file
    ARTICLE_REVISIONS_FILE = json.load(openfile)

# Retrieving and including ORES data for each article
ARTICLE_QUALITY = {}
for article_name in ARTICLE_REVISIONS_FILE.keys():
    
    get_revid = ARTICLE_REVISIONS_FILE[article_name] # Revision ID of article page
    
    # Make ORES request using page title and revision id
    score = request_ores_score_per_article(get_revid)
    unnest_score = score['enwiki']['scores']
    
    for i in unnest_score.values():
        
        # Predicted quality score for specific page
        get_prediction = i['articlequality']['score']
        ARTICLE_QUALITY[article_name] = get_prediction

with open("data/ARTICLE_QUALITY.json", "w") as outfile: 
    json.dump(ARTICLE_QUALITY, outfile, indent = 4) # Write JSON to a file
```




------------------------------------





## Step 3: Combining the Datasets



After retrieving and including the ORES data for each article, we merge the wikipedia and population data together. Consolidate the data into a single CSV file called `wp_politicians_by_country.csv` with the following file schema. 

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>article_title</th>
      <th>article_quality</th>
      <th>revision_id</th>
      <th>country</th>
      <th>population</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shahjahan Noori</td>
      <td>GA</td>
      <td>1099689043</td>
      <td>Afghanistan</td>
      <td>41.1</td>
      <td>SOUTH ASIA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abdul Ghafar Lakanwal</td>
      <td>Start</td>
      <td>943562276</td>
      <td>Afghanistan</td>
      <td>41.1</td>
      <td>SOUTH ASIA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Majah Ha Adrif</td>
      <td>Start</td>
      <td>852404094</td>
      <td>Afghanistan</td>
      <td>41.1</td>
      <td>SOUTH ASIA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Haroon al-Afghani</td>
      <td>B</td>
      <td>1095102390</td>
      <td>Afghanistan</td>
      <td>41.1</td>
      <td>SOUTH ASIA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tayyab Agha</td>
      <td>Start</td>
      <td>1104998382</td>
      <td>Afghanistan</td>
      <td>41.1</td>
      <td>SOUTH ASIA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7451</th>
      <td>Rekayi Tangwena</td>
      <td>Stub</td>
      <td>1073818982</td>
      <td>Zimbabwe</td>
      <td>16.3</td>
      <td>EASTERN AFRICA</td>
    </tr>
    <tr>
      <th>7452</th>
      <td>Josiah Tongogara</td>
      <td>C</td>
      <td>1106932400</td>
      <td>Zimbabwe</td>
      <td>16.3</td>
      <td>EASTERN AFRICA</td>
    </tr>
    <tr>
      <th>7453</th>
      <td>Langton Towungana</td>
      <td>Stub</td>
      <td>904246837</td>
      <td>Zimbabwe</td>
      <td>16.3</td>
      <td>EASTERN AFRICA</td>
    </tr>
    <tr>
      <th>7454</th>
      <td>Herbert Ushewokunze</td>
      <td>Stub</td>
      <td>959111842</td>
      <td>Zimbabwe</td>
      <td>16.3</td>
      <td>EASTERN AFRICA</td>
    </tr>
    <tr>
      <th>7455</th>
      <td>Denis Walker</td>
      <td>C</td>
      <td>1111257734</td>
      <td>Zimbabwe</td>
      <td>16.3</td>
      <td>EASTERN AFRICA</td>
    </tr>
  </tbody>
</table>
<p>7456 rows × 6 columns</p>
</div>




The `population_by_country_2022.csv` contains rows providing cumulative regional population counts. These rows are distinguished by *ALL CAPS* values under the Geography column. Note, a country can only exist in one region: the file represents regions in a *hierarchical order*.


After merging the data, we invariably run into entries which cannot be merged. So we check if population dataset has an entry for equivalent Wikipedia country.



------------------------------------



## Step 4: Analysis

The analysis consists of calculating the following on a country-by-country and regional basis "per capita".


#### `total-articles-per-population`

We get the **total article page count** on the basis for each country (`country-by-country`) and on the basis for each region (`region-by-region`), both stored as a dictionary. To calculate the ratio representing the number of articles per person on a `country-by-country` basis and a `region-by-region` basis, we take the page count and divide it by the total population for each country and each region, respectively. 



#### `high-quality-articles-per-population`

Consider "high quality" articles as those that ORES predicted in either the "FA" (featured article) or "GA" (good article) classes.



----------------------------------------------------



## Step 5: Results

We produce the results from this analysis in the form of six total data tables shown below.


> ### 1. Top 10 countries by coverage:

Below, we get the 10 countries with the `highest total articles` per capita (in descending order).

```
    [('Antigua and Barbuda', 0.00017),
     ('Federated States of Micronesia', 0.00013),
     ('Andorra', 0.0001),
     ('Barbados', 9.333333333333333e-05),
     ('Marshall Islands', 9e-05),
     ('Seychelles', 6e-05),
     ('Montenegro', 5.5e-05),
     ('Luxembourg', 5.2857142857142855e-05),
     ('Bhutan', 5.125e-05),
     ('Grenada', 5e-05)]
```


> ### 2. Bottom 10 countries by coverage:

Next, we get the 10 countries with the `lowest total articles` per capita (in ascending order).



```
    [('Romania', 1.0526315789473685e-07),
     ('Saudi Arabia', 8.174386920980926e-08),
     ('Mexico', 7.843137254901961e-09),
     ('China', 1.3921759710427398e-09),
     ('Liechtenstein', 0),
     ('Monaco', 0),
     ('Nauru', 0),
     ('Palau', 0),
     ('San Marino', 0),
     ('Tuvalu', 0)]
```


> ### 3. Top 10 countries by high quality:

The 10 countries with the highest high quality articles per capita (in descending order).


```
    [('Andorra', 2e-05),
     ('Montenegro', 5e-06),
     ('Albania', 2.1428571428571427e-06),
     ('Suriname', 1.6666666666666667e-06),
     ('Bosnia-Herzegovina', 1.4705882352941177e-06),
     ('Lithuania', 1.0714285714285714e-06),
     ('Croatia', 1.0526315789473683e-06),
     ('Slovenia', 9.523809523809523e-07),
     ('Palestinian Territory', 9.259259259259259e-07),
     ('Gabon', 8.333333333333333e-07)]
```



> ### 4. Bottom 10 countries by high quality: 

The 10 countries with the lowest high quality articles per capita (in ascending order).


```
    [('Sudan', 2.1321961620469082e-08),
     ('Pakistan', 2.1204410517387617e-08),
     ('Uganda', 2.11864406779661e-08),
     ('Colombia', 2.0366598778004074e-08),
     ('Vietnam', 2.012072434607646e-08),
     ('Nigeria', 1.8306636155606407e-08),
     ('Japan', 1.601281024819856e-08),
     ('Thailand', 1.497005988023952e-08),
     ('India', 4.233700254022016e-09),
     ('Tuvalu', 0)]

```



> ### 5. Geographic regions by total coverage:

Now, we get the ranked `ordered list of geographic regions` (in descending order) by total articles per capita.
 


```
    [('SOUTHERN EUROPE', 5.821192052980132e-06),
     ('CARIBBEAN', 4.5454545454545455e-06),
     ('WESTERN EUROPE', 3.49746192893401e-06),
     ('EASTERN EUROPE', 2.5470383275261325e-06),
     ('NORTHERN EUROPE', 2.439252336448598e-06),
     ('WESTERN ASIA', 2.326530612244898e-06),
     ('OCEANIA', 1.9545454545454545e-06),
     ('SOUTHERN AFRICA', 1.6956521739130435e-06),
     ('EASTERN AFRICA', 1.3657505285412261e-06),
     ('SOUTH AMERICA', 1.327188940092166e-06),
     ('WESTERN AFRICA', 1.3186046511627906e-06),
     ('CENTRAL ASIA', 1.3076923076923077e-06),
     ('CENTRAL AMERICA', 1.0786516853932585e-06),
     ('MIDDLE AFRICA', 1.0357142857142857e-06),
     ('NORTHERN AFRICA', 9.043824701195219e-07),
     ('SOUTHEAST ASIA', 6.050295857988165e-07),
     ('SOUTH ASIA', 3.207171314741036e-07),
     ('EAST ASIA', 1.4516129032258064e-07)]
```


> 6. Geographic regions by high quality coverage:

Rank ordered list of geographic regions (in descending order) by high quality articles per capita.


```
    [('SOUTHERN EUROPE', 5.821192052980132e-06),
     ('CARIBBEAN', 4.5454545454545455e-06),
     ('WESTERN EUROPE', 3.49746192893401e-06),
     ('EASTERN EUROPE', 2.5470383275261325e-06),
     ('NORTHERN EUROPE', 2.439252336448598e-06),
     ('WESTERN ASIA', 2.326530612244898e-06),
     ('OCEANIA', 1.9545454545454545e-06),
     ('SOUTHERN AFRICA', 1.6956521739130435e-06),
     ('EASTERN AFRICA', 1.3657505285412261e-06),
     ('SOUTH AMERICA', 1.327188940092166e-06),
     ('WESTERN AFRICA', 1.3186046511627906e-06),
     ('CENTRAL ASIA', 1.3076923076923077e-06),
     ('CENTRAL AMERICA', 1.0786516853932585e-06),
     ('MIDDLE AFRICA', 1.0357142857142857e-06),
     ('NORTHERN AFRICA', 9.043824701195219e-07),
     ('SOUTHEAST ASIA', 6.050295857988165e-07),
     ('SOUTH ASIA', 3.207171314741036e-07),
     ('EAST ASIA', 1.4516129032258064e-07)]


```

------------------------------------




## Research Implications


Reflect on what you have learned, what you found, what (if anything) surprised you about your findings, and/or what theories you have about why any biases might exist (if you find they exist). 


1. What biases did you expect to find in the data (before you started working with it), and why?


2. What (potential) sources of bias did you discover in the course of your data processing and analysis?

3. What might your results suggest about (English) Wikipedia as a data source?

4. What might your results suggest about the internet and global society in general?
5. Can you think of a realistic data science research situation where using these data (to train a model, perform a hypothesis-driven research, or make business decisions) might create biased or misleading results, due to the inherent gaps and limitations of the data?
6. Can you think of a realistic data science research situation where using these data (to train a model, perform a hypothesis-driven research, or make business decisions) might still be appropriate and useful, despite its inherent limitations and biases?
How might a researcher supplement or transform this dataset to potentially correct for the limitations/biases you observed?




