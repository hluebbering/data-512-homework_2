[![Typing SVG](https://readme-typing-svg.demolab.com?font=Roboto+Black&size=38&duration=2800&pause=2000&color=417649&background=FFFFFF00&vCenter=true&width=670&lines=Homework+2.+Considering+Bias+in+Data)](https://git.io/typing-svg)


[![GitHub WidgetBox](https://github-widgetbox.vercel.app/api/skills?names=js,ts,java,php,python,html,css,c,cpp,csharp,swift,rust,ruby,kotlin,erlang,dart,go,scala,elm,bash,r,xml,json,yaml,postgresql,mysql,haskell,powershell,lua,visualbasic,x86,arm,groovy,perl,solidity,fortran,sass,graphql,clojure,clojurescript,markdown)](https://github.com/Jurredr/github-widgetbox)


[![GitHub WidgetBox](https://github-widgetbox.vercel.app/api/profile?username=hluebbering&data=followers,repositories,stars,commits)](https://github.com/hluebbering/github-widgetbox)


[![spotify-github-profile](https://spotify-github-profile.vercel.app/api/view?uid=hannahluebbering&cover_image=true&theme=novatorem&show_offline=false&bar_color=d528e2&bar_color_cover=false)](https://spotify-github-profile.vercel.app/api/view?uid=hannahluebbering&redirect=true)

[![GitHub Streak](http://github-readme-streak-stats.herokuapp.com?user=hluebbering&theme=gruvbox&border_radius=12&date_format=M%20j%5B%2C%20Y%5D&fire=FF7500&sideNums=FF7500&dates=A2BD7F&stroke=7D1D40&currStreakNum=FF1578&currStreakLabel=FF1578&sideLabels=FF81E2)](https://git.io/streak-stats)

<link href="https://raw.githubusercontent.com/hluebbering/data-512-homework_2/main/data/custom.css" rel="stylesheet"></link>


```python
from IPython.display import HTML, display
from urllib.request import urlopen

CSS_URL = "https://raw.githubusercontent.com/rsomani95/jupyter-custom-theme/master/custom.css"
CSS_URL = "https://raw.githubusercontent.com/hluebbering/data-512-homework_2/main/data/custom.css"
CSS = urlopen(CSS_URL)
CSS = CSS.read().decode('utf-8')
HTML_CSS = f"""
<style>
{CSS}
</style>
"""
HTML(HTML_CSS)
```





<style>
 .markdown-body h2 {
    padding-bottom: .3em;
    font-size: 1.5em;
    border-bottom: 1px solid var(--color-border-muted);
    background: purple !important;
}
body {
  font-family: Roboto;

}

h1 {
    font-weight: 800; 
    font-family: Roboto; 
    color: black;
}

h2 {
    color: white;
    font-family: Roboto Condensed;
    text-shadow: 0.5pt 0.5pt 0.5pt black, 1pt 1pt 1pt black, -0.25pt -0.435pt 0.35pt hsl(0deg 0% 100% / 50%);
    filter: drop-shadow(0.5px 0.5px 0.5px hsl(176deg 65% 10% / 75%));
    padding: 6pt 4pt;
    background: hsl(180deg 25% 15%);
    background-image: radial-gradient(teal 20%, transparent 0), radial-gradient(teal 20%, transparent 0);
    background-size: 30px 30px;
    background-position: 0 0, 15px 15px;
    letter-spacing: 0.25pt;
    display: inline-flex;
    border-radius: 6pt;
}

.myfont {
    background: azure;
    color: hsl(219deg 90% 58%);
    text-shadow: 0.125pt 0.375pt 0.45pt hsl(208deg 100% 40% / 95%);
    font-weight: 400;
    font-family: Roboto;
    border-radius: 4pt;
    padding: 2pt 3pt;
    box-shadow: 0.5pt 0.5pt 1.125pt #333;
}
mark {
    background:lavender;
    color:black;
    font-weight: 700;
    border-radius: 5pt;
    padding: 2pt 3pt;
}

mark.mark2, mark.mark3 {
    background: hsl(120deg 25% 75%);
    font-size: 12pt;
    box-shadow: 0.5pt 0.5pt 1pt hsl(0deg 0% 47% / 75%);
    text-shadow: 0.125pt 0.35pt 1pt hsl(0deg 0% 0% / 92%);
    color: white;
    padding: 3pt 6pt;
    font-weight: 800;
    font-size: 11.5pt;
    letter-spacing: 0.125pt;
}

mark.mark3 {
    background: hsl(170deg 45% 75%);
    font-size: 11pt;
    padding: 2pt 4pt;
}

code.mycode {
    background: lavender;
    padding: 2pt 4pt;
    border-radius: 4pt;
    box-shadow: 0.5pt 0.5pt 1.5pt hsl(0deg 0% 34% / 75%);
}

code.mycode2 {
    background: hsl(208deg 100% 97% / 50%);
    font-family: Roboto Condensed;
    color: darkslategray;
    padding: 3pt 4pt;
    border-radius: 4pt;
    box-shadow: 0.5pt 0.5pt 1.5pt hsl(0deg 0% 34% / 75%);
    font-size: 9pt;
    line-height: 2.5;
}

</style>



 
# Homework 2. Considering Bias in Data 



<mark>Project Goal:</mark> The following explores the concept of data bias using Wikipedia articles considering political figures from different countries. For this project, we combine a dataset of Wikipedia articles with a dataset of country populations. We then use `ORES`, a machine learning service, to estimate the quality of each article.

We perform an analysis of how the coverage of politicians on Wikipedia and the quality of articles about politicians varies among countries. The analysis consists of a series of tables that show the following:

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




<style>
body {
  font-family: Roboto;

}

h1 {
    font-weight: 800; 
    font-family: Roboto; 
    color: black;
}

h2 {
    color: white;
    font-family: Roboto Condensed;
    text-shadow: 0.5pt 0.5pt 0.5pt black, 1pt 1pt 1pt black, -0.25pt -0.435pt 0.35pt hsl(0deg 0% 100% / 50%);
    filter: drop-shadow(0.5px 0.5px 0.5px hsl(176deg 65% 10% / 75%));
    padding: 6pt 4pt;
    background: hsl(180deg 25% 15%);
    background-image: radial-gradient(teal 20%, transparent 0), radial-gradient(teal 20%, transparent 0);
    background-size: 30px 30px;
    background-position: 0 0, 15px 15px;
    letter-spacing: 0.25pt;
    display: inline-flex;
    border-radius: 6pt;
}

.myfont {
    background: azure;
    color: hsl(219deg 90% 58%);
    text-shadow: 0.125pt 0.375pt 0.45pt hsl(208deg 100% 40% / 95%);
    font-weight: 400;
    font-family: Roboto;
    border-radius: 4pt;
    padding: 2pt 3pt;
    box-shadow: 0.5pt 0.5pt 1.125pt #333;
}
mark {
    background:lavender;
    color:black;
    font-weight: 700;
    border-radius: 5pt;
    padding: 2pt 3pt;
}

mark.mark2, mark.mark3 {
    background: hsl(120deg 25% 75%);
    font-size: 12pt;
    box-shadow: 0.5pt 0.5pt 1pt hsl(0deg 0% 47% / 75%);
    text-shadow: 0.125pt 0.35pt 1pt hsl(0deg 0% 0% / 92%);
    color: white;
    padding: 3pt 6pt;
    font-weight: 800;
    font-size: 11.5pt;
    letter-spacing: 0.125pt;
}

mark.mark3 {
    background: hsl(170deg 45% 75%);
    font-size: 11pt;
    padding: 2pt 4pt;
}

code.mycode {
    background: lavender;
    padding: 2pt 4pt;
    border-radius: 4pt;
    box-shadow: 0.5pt 0.5pt 1.5pt hsl(0deg 0% 34% / 75%);
}

code.mycode2 {
    background: hsl(208deg 100% 97% / 50%);
    font-family: Roboto Condensed;
    color: darkslategray;
    padding: 3pt 4pt;
    border-radius: 4pt;
    box-shadow: 0.5pt 0.5pt 1.5pt hsl(0deg 0% 34% / 75%);
    font-size: 9pt;
    line-height: 2.5;
}



</style>



----------------------------------------------------


## Step 1: Getting the Article and Population Data



<div class="alert alert-block alert-info">

We get data that lists <span class = "myfont">Wikipedia articles of politicians</span> and data for <span class = "myfont">country populations</span>.

</div>

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



## Step 2: Getting Article Quality Predictions

<div class="alert alert-block alert-info">

Now we get the `predicted quality scores` for each article in the Wikipedia dataset using `ORES`.
    
</div>


- ORES is a machine learning tool that provides estimates of Wikipedia article quality. 
- The article quality estimates are, from best to worst:
    - `FA:` Featured article
    - `GA:` Good article
    - `B:` B-class article
    - `C:` C-class article
    - `Start:` Start-class article
    - `Stub:` Stub-class article


<span style="color: pink !important;">hey</span>

<div class="alert alert-block alert-warning">

1. To get a Wikipedia page quality prediction from ORES for each politician’s article page: 
    1. Read each line of `politicians_by_country.SEPT.2022.csv`
    2. Make a <code class ="mycode2">page info request</code> to get the current page revision id 
    3. Make an <code class ="mycode2">ORES request</code> using the *page title* and *current revision id*
</div>


```python
# List of Wikipedia article pages about politicians
politician_names = []
for name in politician_articles['name']:
    politician_names.append(name)
```

<mark class="mark2">MediaWiki API: Making Page Info Request</mark>

ORES requires a specific revision ID of a specific article to be able to make a label prediction. We use the [API:Info](https://www.mediawiki.org/wiki/API:Info) request to get a range of metadata on an article, including the most current revision ID of the article page. The following code illustrates how to access page info data using the `MediaWiki REST API` for the EN Wikipedia.



```python
# Import python modules
import json, time, urllib.parse, requests

#########
# CONSTANTS

API_ENWIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"
API_LATENCY_ASSUMED = 0.002
API_THROTTLE_WAIT = (1.0/100.0)-API_LATENCY_ASSUMED

REQUEST_HEADERS = {
    'User-Agent': 'luebhr@uw.edu, University of Washington, MSDS DATA 512 - AUTUMN 2022',
}

#PAGEINFO_EXTENDED_PROPERTIES = "talkid|url|watched|watchers"
PAGEINFO_EXTENDED_PROPERTIES = ""

PAGEINFO_PARAMS_TEMPLATE = {
    "action": "query",
    "format": "json",
    "titles": "",           # to simplify this should be a single page title at a time
    "prop": "info",
    "inprop": PAGEINFO_EXTENDED_PROPERTIES
}

```


```python
#########
# PROCEDURES/FUNCTIONS
def request_pageinfo_per_article(article_title = None, 
                                 endpoint_url = API_ENWIKIPEDIA_ENDPOINT, 
                                 request_template = PAGEINFO_PARAMS_TEMPLATE,
                                 headers = REQUEST_HEADERS):
    if not article_title: return None 
    request_template['titles'] = article_title
    
    try:
        if API_THROTTLE_WAIT > 0.0:
            time.sleep(API_THROTTLE_WAIT)
        response = requests.get(endpoint_url, headers=headers, params=request_template)
        json_response = response.json()
    except Exception as e:
        print(e)
        json_response = None
    return json_response
```


For each Wikipedia article from our list of article pages about politicians, we make a page info request to get the current page revision id used for ORES scoring. We then save the article revisions dictionary to JSON file.



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

*Note: Some articles have no score. Below is a log of articles for which we were not able to retrieve an ORES score.*



```python
print("Articles with no retrievable ORES score:")
article_without_ORES_score
```

    Articles with no retrievable ORES score:
    




    ['Prince Ofosu Sefah',
     'Harjit Kaur Talwandi',
     'Abd al-Razzaq al-Hasani',
     'Kang Sun-nam',
     'Abiodun Abimbola Orekoya',
     'Segun “Aeroland” Adewale',
     'Roman Konoplev',
     'Nhlanhla “Lux” Dlamini']



<mark class = "mark2">Scores API: Making an ORES Request</mark>


This example illustrates how to generate quality scores for article revisions using ORES. This example shows how to request a score of a specific revision, where the score provides probabilities for all of the possible article quality levels. The API documentation can be access from the main ORES page.



```python
#########
# CONSTANTS

# The current ORES API endpoint
API_ORES_SCORE_ENDPOINT = "https://ores.wikimedia.org/v3"
# A template for mapping to the URL
API_ORES_SCORE_PARAMS = "/scores/{context}/{revid}/{model}"

# Use some delays so that we do not hammer the API with our requests
API_LATENCY_ASSUMED = 0.002
API_THROTTLE_WAIT = (1.0/100.0)-API_LATENCY_ASSUMED
REQUEST_HEADERS = {
    'User-Agent': 'luebhr@uw.edu, University of Washington, MSDS DATA 512 - AUTUMN 2022'
}

# This template lists the basic parameters for making an ORES request
ORES_PARAMS_TEMPLATE = {
    "context": "enwiki",        # which WMF project for the specified revid
    "revid" : "",               # the revision to be scored - this will probably change each call
    "model": "articlequality"   # the AI/ML scoring model to apply to the reviewion
}
```

The API request will be made using one procedure. The idea is to make this reusable. The procedure is parameterized, but relies on the constants above for the important parameters. The underlying assumption is that this will be used to request data for a set of article revisions. Therefore, the main parameter is article_revid.


```python
#########
# PROCEDURES/FUNCTIONS

def request_ores_score_per_article(article_revid = None, 
                                   endpoint_url = API_ORES_SCORE_ENDPOINT, 
                                   endpoint_params = API_ORES_SCORE_PARAMS, 
                                   request_template = ORES_PARAMS_TEMPLATE,
                                   headers = REQUEST_HEADERS,
                                   features=False):
    # Make sure we have an article revision id
    if not article_revid: return None
    
    # set the revision id into the template
    request_template['revid'] = article_revid
    
    # Combine endpoint_url with parameters for request URL
    request_url = endpoint_url+endpoint_params.format(**request_template)
    
    # Features used by ML model can sometimes be returned as well as scores
    if features:
        request_url = request_url+"?features=true"
    
    # make the request
    try:
        if API_THROTTLE_WAIT > 0.0:
            time.sleep(API_THROTTLE_WAIT)
        response = requests.get(request_url, headers=headers)
        json_response = response.json()
    except Exception as e:
        print(e)
        json_response = None
    return json_response
```

For each article page, we make an ORES request using the <code class ="mycode">page title</code> and current <code class ="mycode">revision id</code>. We saved this information in the `ARTICLE_REVISIONS.json` file. 



```python
# Open ARTICLE_REVISIONS file
with open('data/ARTICLE_REVISIONS.json', 'r') as openfile:
    
    # Read from json file
    ARTICLE_REVISIONS_FILE = json.load(openfile)


```


```python
# Retrieving and including ORES data for each article
ARTICLE_QUALITY = {}

for article_name in ARTICLE_REVISIONS_FILE.keys():
    
    # Revision ID of article page
    get_revid = ARTICLE_REVISIONS_FILE[article_name]
    
    # Make ORES request using page title and revision id
    score = request_ores_score_per_article(get_revid)
    unnest_score = score['enwiki']['scores']
    
    for i in unnest_score.values():
        
        # Predicted quality score for specific page
        get_prediction = i['articlequality']['score']
        ARTICLE_QUALITY[article_name] = get_prediction


# Write JSON to a file
with open("data/ARTICLE_QUALITY.json", "w") as outfile:
    json.dump(ARTICLE_QUALITY, outfile, indent = 4)
```

<br>


----------------------------------------------------




## Step 3: Combining the Datasets


After retrieving and including the ORES data for each article, we merge the wikipedia and population data together. Consolidate the data into a single CSV file called `wp_politicians_by_country.csv` with the following file schema.

![](schema.png)

The `population_by_country_2022.csv` contains rows providing <code class ="mycode">cumulative regional population counts</code>. These rows are distinguished by *ALL CAPS* values under the <code class ="mycode">Geography</code> column. Note, a country can only exist in one region: the file represents regions in a *hierarchical order*.



```python
region_dict = {}
region_pop = {} # Cumulative regional population counts
country_pop = {} # Cumulative country population counts

# Iterate through each row
for index, row in country_populations.iterrows():
    
    if row['Geography'].isupper():
        
        GET_REGION = row['Geography']
        GET_REGION_POP = row['Population (millions)']
        REGION_COUNTRIES = []
        
        region_dict[GET_REGION] = REGION_COUNTRIES
        region_pop[GET_REGION] = GET_REGION_POP
    
    else:
        GET_COUNTRY = row['Geography']
        REGION_COUNTRIES.append(GET_COUNTRY)
        
        country_pop[GET_COUNTRY] = float(row['Population (millions)'])
 
```


```python
country_by_region = [] # Match country to region

for key, vals in region_dict.items():
    for i in vals:
        country_dict = {}
        
        country_dict['country'] = i
        country_dict['region'] = key
        
        for j in country_pop:
            if j == i:
                country_dict['population'] = country_pop[j]
                
        country_by_region.append(country_dict)

```

<br>

Below we merge the predicted quality scores for each politician article page and the respective politician's country. We saved this information in the `ARTICLE_QUALITY.json` file.


```python
# Open ARTICLE_QUALITY file
with open('data/ARTICLE_QUALITY.json', 'r') as openfile:
    ARTICLE_QUALITY_FILE = json.load(openfile)
```


```python
# Convert pandas dataframe to dictionary
politician_articles_dict = politician_articles.to_dict(orient = 'records')
page_data = [] # Merge wikipedia and population data

for i in ARTICLE_QUALITY_FILE.items():
    info_dict = {} # Page information
    get_title = i[0] # Page title
    get_quality = i[1] # Retrieved ORES score
    
    # Add predicted quality score for each article
    info_dict['article_title'] = get_title
    info_dict['article_quality'] = get_quality['prediction']
    
    # Get revision ID for article
    for j in ARTICLE_REVISIONS_FILE.items():
        if j[0] == get_title:
            info_dict['revision_id'] = j[1]
            
    # Get country name for particular politician
    for x in politician_articles_dict:
        if x['name'] == get_title:
            info_dict['country'] = x['country']        
      
    # Store all information for each page       
    page_data.append(info_dict)
```

<br>

After merging the data, we invariably run into entries which cannot be merged. So we check if population dataset has an entry for equivalent Wikipedia country, or vice-versa as follows.



```python
all_countries = [] # Countries in population dataset

for i in country_by_region:
    all_countries.append(i['country'])
```


```python
no_match = [] # Wikipedia countries with no matches
new_page_data = [] # Updated page data

for i in range(len(page_data)):
    
    get_dict = page_data[i]
    
    
    # Countries with matched entry
    if get_dict['country'] in all_countries:
        
        # Get population for particular country
        for j in country_by_region:
            if j['country'] == get_dict['country']:
                get_dict['population'] = j['population']
                get_dict['region'] = j['region']
                
        
        new_page_data.append(get_dict)
        
    # Entries that cannot be merged
    else:
        no_match.append(get_dict['country'])

```


```python
len(new_page_data)
```




    7456



Identify all countries for which there are no matches and output a list of those countries, with each country on a separate line called: `wp_countries-no_match.txt`


```python
# Output list to text file
with open('wp_countries-no_match.txt', 'w') as f:
    for line in no_match:
        f.write(f"{line}\n")
```

Consolidate the rest of the data into a single CSV file called `wp_politicians_by_country.csv`.


```python
# Convert list of dictionaries
df = pd.DataFrame.from_dict(new_page_data)
df.to_csv('wp_politicians_by_country.csv', index = False)

df
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



----------------------------------------------------



## Step 4: Analysis

The analysis consists of calculating the following on a country-by-country and regional basis:

- <code class = "mycode2">total-articles-per-population</code> - a ratio representing the number of articles per person
- <code class = "mycode2">high-quality-articles-per-population</code> - a ratio representing the number of high quality articles per person

All of these values are to be "per capita".

-  For your analysis always put a country in the closest (lowest in the hierarchy) region.
- Keep in mind that `population_by_country_2022.csv` provides population in millions. The calculated proportions in this step are likely to be very small numbers.

#### `total-articles-per-population`

Below, we get the **total article page count** on the basis for each country (<code class = "mycode">country-by-country</code>) and on the basis for each region (<code class = "mycode">region-by-region</code>), both stored as a dictionary.



```python
articles_per_country = {} # country-by-country basis
articles_per_region = {} # region-by-region basis

# iterate through each row
for index, row in df.iterrows():
    
    get_country = row['country']
    get_region = row['region']
    
    
    # Page count by country
    if get_country in articles_per_country.keys():
        articles_per_country[get_country] += 1
    else:
        articles_per_country[get_country] = 1
     
    
    # Page count by region
    if get_region in articles_per_region.keys():
        articles_per_region[get_region] += 1
    else:
        articles_per_region[get_region] = 1

```

To calculate the ratio representing the number of articles per person on a `country-by-country` basis and a `region-by-region` basis, we take the page count and divide it by the total population for each country and each region, respectively. 


```python
articles_per_country_ratio = {} # country-by-country basis

for i, j in articles_per_country.items():
    
    country = i
    page_count = j
    
    for x, y in country_pop.items():
        if x == country:
            get_population = y * 1000000
            
            if get_population == 0:
                get_ratio = 0
                articles_per_country_ratio[country] = get_ratio
                
            else:
                get_ratio = page_count / get_population
                articles_per_country_ratio[country] = get_ratio
                               

```


```python
articles_per_region_ratio = {} # region-by-region basis

for i, j in articles_per_region.items():
    
    region = i
    page_count = j
    
    for x, y in region_pop.items():
        if x == region:
            region_population = y * 1000000
            
            if region_population == 0:
                region_ratio = 0
                articles_per_region_ratio[region] = region_ratio
                
            else:
                region_ratio = page_count / region_population
                articles_per_region_ratio[region] = region_ratio
                

```

#### `high-quality-articles-per-population`

Consider "high quality" articles as those that ORES predicted in either the "FA" (featured article) or "GA" (good article) classes.



```python
# country-by-country basis

# regional basis
```

----------------------------------------------------



## Step 5: Results

We produce the results from this analysis in the form of six total data tables that show:

1. `Top 10 countries by coverage:` The 10 countries with the highest total articles per capita (in descending order) .
2. `Bottom 10 countries by coverage:` The 10 countries with the lowest total articles per capita (in ascending order) .
3. Top 10 countries by high quality: The 10 countries with the highest high quality articles per capita (in descending order).
4. Bottom 10 countries by high quality: The 10 countries with the lowest high quality articles per capita (in ascending order).
5. `Geographic regions by total coverage:` A rank ordered list of geographic regions (in descending order) by total articles per capita.
6. Geographic regions by high quality coverage: Rank ordered list of geographic regions (in descending order) by high quality articles per capita.




1. <mark class = "mark3">Top 10 countries by coverage:</mark>

Below, we get the 10 countries with the <code class="mycode2">highest total articles</code> per capita (in descending order).


```python
# Sort by value
sort_country_ratio = sorted(
    articles_per_country_ratio.items(),
    key=lambda item: item[1], reverse = True)

sort_country_ratio[0:10]
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




<br>

2. <mark class = "mark3">Bottom 10 countries by coverage:</mark>

Next, we get the 10 countries with the <code class="mycode2">lowest total articles</code> per capita (in ascending order).



```python
n = len(sort_country_ratio) 
sort_country_ratio[n-10:n]
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



3. <mark class = "mark3">Top 10 countries by high quality:</mark> 

The 10 countries with the highest high quality articles per capita (in descending order).


```python

```

4. <mark class = "mark3">Bottom 10 countries by high quality:</mark> The 10 countries with the lowest high quality articles per capita (in ascending order).


```python

```

------

<br>

5. <mark class = "mark3">Geographic regions by total coverage:</mark>

Now, we get the ranked <code class="mycode2">ordered list of geographic regions</code> (in descending order) by total articles per capita.
 



```python
# Sort by value
sort_region_ratio = sorted(
    articles_per_region_ratio.items(),
    key=lambda item: item[1], reverse = True)

sort_region_ratio
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



6. <mark class = "mark3">Geographic regions by high quality coverage:</mark> 

Rank ordered list of geographic regions (in descending order) by high quality articles per capita.


```python

```
