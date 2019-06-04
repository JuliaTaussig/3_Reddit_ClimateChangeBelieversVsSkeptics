## Project 3: Predicting Whether Reddit Posts from Climate Change Believers vs. Climate Change Skeptics Subreddits

### Background Information: 
According to digitaltrends.com, Reddit can be boasted as the “front page of the internet.” As of July of 2018, Reddit was the fifth most popular site in the United States, according to Alexa, and the 18th worldwide.  Reddit is a large collection of forums and acts as a place where people can share news and content and/or comment on other people’s posts (Nicol, Will, https://www.digitaltrends.com/web/what-is-reddit/).  Two subreddits (which are specific forums within Reddit) were selected for analysis during this project: r/climate and r/climateskeptics.  The r/climate subreddit defines itself as "a community for truthful science-based news about climate and related politics and activism" (see the community details section of https://www.reddit.com/r/climate).  The r/climateskeptics subreddit defines itself as a place for "questioning climate related environmentalism" ((see the community details section of https://www.reddit.com/r/climateskeptics).  It is assumed that typically global climate change believers post on the r/climate subreddit, and typically global climate change skeptics post on the r/climateskeptics subreddit.

### Problem Statement/Goal: 
The goals of the project are to:  
- Use Reddit's API to collect posts from the following subreddits: r/climate and r/climateskeptics.  
- Binary classification problem: Use natural language processing (NLP) to train a classifier on which subreddit a given post came from (accuracy of the classifiers will also be evaluated). 

### Project Outline
1. Used Reddit's API to collect posts from the following subreddits: r/climate and r/climateskeptics.
2. Imported subreddit post data into pandas dataframes
3. Cleaned the data (with a focus on null values)
4. Performed exploratory data analysis (EDA) to find relationships between features (words/terms) and the target variable (0 if the post came from r/climate (believers) and 1 if the post came from r/climateskeptics (skeptics)) and to gain other insights
5. Created a highly over-simplified baseline model to beat
6. Created a Naive Bayes classifiaction model to predict which subreddit a given post came from and assessed its accuracy scores when compared to the baseline model
7. Created a logistic classifiaction model to predict which subreddit a given post came from and assessed its accuracy scores when compared to the baseline model (also singular-value decomposition (SVD) was used to fit and transform training data and to transform testing data and GridSearchCV were used to work toward optimizing the logistic regression model). 
8. Analyzed model parameters and accuracy to see what the most important features were, and presented findings to a semi-technical audience which could include stakeholders (such as a team supporting a political party, a scientific organization focused on increasing global climate change awareness, or another company interested in gaining insights from textual data)

### Data Dictionary

The dataframe called df had the following variables:  

|Feature|Type|Description|
|---|---|---|---|
|title|*object*|subreddit post titles (before preprocessed)|
|full_post|object|subreddit full post (in json format for future unpacking)|
|permalink|object|subreddit permalink for future reference (especially to locate comments related to the post)|
|post_text|object|subreddit post text (which was sometimes included following the title)|
|full_comments_page_each_post|object|subreddit full comments post (in json format for future unpacking)|
|skeptic|integer|the target variable: values were 0 if the post was from the r/climate subreddit, and values were 1 if the post was from the r/climateskeptics subreddit|
|char_count|integer|the number of characters in the subreddit post title (before title cleaning/preprocessing)|
|word_count|integer|the number of words in the subreddit post title (before title cleaning/preprocessing)|
|title_preprocessed|object|subreddit post title after clearning/preprocessing|
                      
### Repository Structure

#### Contents of project_3 folder: 
- Folder: Project3_JuliaTaussig_DEN
- README.md (markdown file from General Assembly)
- Requirements.txt (text file from General Assembly)

#### Contents of Project3_JuliaTaussig_DEN folder:
- folder: code 
- folder: data 
- README.md (this markdown file)
- Project3_Presn_JuliaTaussig.pdf: presentation outlining my process and findings for a semi-technical audience 

#### Contents of code folder:
- Jupyter Notebook: 1_Project3_JuliaTaussig_GatheringData.ipynb
- Jupyter Notebook: 2_Project3_JuliaTaussig_Cleaning_EDA_ModelBuilding_Analysis.ipynb

#### Contents of data frolder:
Files that were not used for analysis due to formatting complications (post comments were not collected in these datasets due to challenges with collecting post comments, so these datasets did not contain all of the information that was thought to be needed for analysis and therefore were not used for analysis):  
- climateskeptics_post_titles_20190328_0850.csv
- climate_post_titles_20190328_0850.csv
- climate_post_titles_20190328_0915.csv
- climateskeptics_post_titles_20190328_0915.csv
- climate_posts_20190328_0915.csv
- climate_posts_with_titles_20190328_0915.csv
- skeptics_posts_with_titles_20190328_0915.csv
- climate_posts_with_titles_20190329_0545.csv
- skeptics_posts_with_titles_20190329_0545.csv

Files that were used for analysis (these files contained all of the desired information for analysis, including post comments):  
- climate_posts_detailed_20190401_0955.csv
- skeptics_posts_detailed_20190401_1040.csv
- climate_posts_detailed_20190402_2100.csv
- skeptics_posts_detailed_20190403_0640.csv  

Note: Files were saved using year month day then 24-hour time convention to keep track of times when the data was saved.

### Executive Summary

#### Data collection
Reddit's API along with the request library were used to collect posts from the following subreddits: r/climate and r/climateskeptics.  There were difficulties with acquiring comments, and eventually the code worked.  This meant that the number of columns in the data collected kept on changing.  The data collected on 4/01/2019 to 4/03/2019 had the same structure, so that data was used (since there was concern that the other datasets would have a lot of null values and could generate bias in analysis, if the post comments had been used in this analysis).  If there was more time, it would have been great to collect more data using this structure.

#### Data Importing and EDA and Cleaning
After importing the four subreddit datasets that were decided to be used for analysis into pandas dataframes, data was cleaned and inspected.  A column called "Unnamed: 0" was removed from each dataframe (this column was present in each dataframe likely due to a mistake during data exporting after data collection or during data importing).  A column called "skeptic" was added to all four dataframes to show where each subreddit post came from.  A value of 0 meant that the post came from the r/climate subreddit while a value of 1 meant that the post came from the r/climateskeptics subreddit.  All four dataframes were combined into a dataframe called df, duplicate values were removed, and index values were reset.  Null values were then inspected, and they were all in the post_text column.  Most of the subreddit posts did not have post text, so any null values in post_text were replaced with empty strings in case the subreddit post text would be analyzed later.  The dataframe columns' datatypes were mostly objects, and the skeptic column was filled with integers (0 or 1), as expected.  

The target variable (skeptics) value counts were inspected.  There were 1002 r/climateskeptics posts and 995 r/climate posts, so approx. 50.2% of the subreddit posts were from r/climateskeptics, and approx. 49.8% of the subreddit posts were from r/climate.  There is a pretty good balance in the target class.  

The baseline model was predicting that every post was in the majority class, meaning the post was in the r/climateskeptics subreddit page. The accuracy of the baseline model was approx. 0.502 (meaning 50.2% of predictions would be correct).  The goal was to beat this baseline model accuracy of 0.502.

Total character counts for subreddit titles were investigated and plotted. It was found that skeptics more often had shorter post titles (based on number of characters); therefore, climate believers were more likely to have longer post titles.  Upper outliers appeared to be more likely believers.  The believers and skeptics distributions (of post title character length) were both skewed to the right, but there was a larger skew to the right for the believers distribution. 

Total word counts for subreddit titles were investigated and plotted.  It was found that skeptics more often had shorter post titles (based on number of words); therefore, climate believers were more likely to have longer post titles.  Upper outliers appeared to be more likely believers.  The believers and skeptics distributions (of post title word count) were both skewed to the right, but there was a larger skew to the right for the believers distribution. 

Subreddit post titles were then preprocessed.  The titles were changed to all lower-case, punctuation was removed, the words/terms were tokenized (split), the words/terms were lemmatized, English stopwords were removed, and the post titles were joined back into strings.  The preprocessed titles were inspected and then were placed in a "title_preprocessed" column in the dataframe.

The preprocessed titles were then CountVectorized and placed in a term matrix called term_matrix and then a dataframe called term_df.  A column called our_target was added to term_df (filled with the target class information).  Term counts were then investigated, and the top 15 and then 16 to 30 most frequent words (excluding stop words) were visualized using heat maps. The 10 most correlated terms with our_target=0 (believer posts) and the 10 most correlated terms with our_target=1 (skeptic posts) were inspected. The groupby method was used to get some aggregates over the classes, and sum was used to find total times a word occurred in a class, and mean was used to find average times words occurred in classes.  The terms that were most freuqently used in believer posts and skeptic posts  were inspected.  Terms with the highest ratio of use in skeptic posts vs. believer posts and terms with the highest ratio of use in believer posts vs. skeptic posts were inspected.  

A hypothsis test was conducted on the words that overlapped in the top 40 words in each class (highest term frequency relative to the target class: top 40 words that appeared most in r/climateskeptic subreddit post titles and top 40 words that appeared most in r/climate subreddit post titles).  This was done because when lemmatizing, because of the aggressive nature of conversion, artificial overlap (words that had wholly different meanings) could be lemmatized down to the same lem.  The null hypothesis was that the subreddits for global climate change skeptics and believers have the same mean frequency for word x.  Several words could not reject the null hypothes, and several words rejected the null hypothesis (the list of rejected null hypothesis words was larger than the list of not rejected null hypothesis words).  The distributions of terms that did not occur with the same frequency in both subreddits (rejected null hypothesis) were plotted to see which subreddits they occurred more frequently in.   

The maximum, mean, and sum term frequency*inverse document frequency (TF-IDF) of terms were then inspected.  Sentiment analysis was conducted by class, and it was found that global climate change skeptic post titles were slightly more negative, less neutral, and less positive than believer posts.  The difference in sentiment was smaller than expected.  Parts of speech of terms by class were also analyzed.  It's interesting that there were more verbs, nouns, interjections, and total words with easily identified parts of speech in the r/climate posts than the r/climateskeptics post titles.  Both subreddits had similar numbers of adjectives.

#### Modeling

The data was train-test-split, and then the words in the train set were fit and transformed by a count vectorizer while the test set were transformed by the count vectorizer. 

A Naive Bayes classifiaction model was created to predict which subreddit a given post came from, and its accuracy scores surpassed the baseline accuracy of 0.502.  The train score was much better than the test score and cross-validation score.  This means that the model was overfit to the training data (it suffered from high variance).  A different model was then created to determine whether it was possible that changing the model type would result in a lower model variance. 

A logistic classifiaction model was created to predict which subreddit a given post came from, and its accuracy scores surpassed the baseline accuracy of 0.502.  The train score increased significantly while the test and cross-validation scores did not increase significantly, so the model became more overfit.

Singular-value decomposition (SVD) was used to fit and transform training data and to transform testing data to see if certain groups of words (forming components) were more important than others and to reduce affects of less important groupings of words.  Some EDA was conducted during set-up of the SVD as well.  Logistic regression of the SVD-transformed data resulted in a lower train score and a slightly higher test score and therefore a less overfit model, but the model accuracy scores did not increase as significantly as desired.

I'm unsure about whether I conducted SVD correctly.  I think I should have conducted the TF-IDF vectorization fit and transformation and the SVD fit and transformation  only on the training set and then only completed the TF-IDF vectorization transformation and SVD transformation on the test set.  I intend to increase my understanding of the SVD process.

GridSearchCV was used to work toward optimizing hyperparameters of the logistic regression model of SVD-transformed data, but there was no significant improvement on the train, test, and cross-validation scores.   

It would have been great to collect more post titles and to analyze post text and post comment text and other features of subreddit posts such as post authors, popularity of posts, and emogis used.  It would have been interesting to optimize the train-test-split of the train and test sets.  It also would have been interesting to try different models on optimized count vectorized and TF-IDF vectorized data (it would be great to see what min_df, max_df and other parameters should be used to determine which terms (features) are most important/useful) to improve model accuracy and to reduce model variance.  For example, boosted tree models and random forest models could reduce model variance.  It would be interesting to use a k-nearest neighbors model, decision trees, and bagged tree models to see which model produces the best train, test, and cross-validation accuracy scores.  It would also be interesting to try ngram=2 in count vectorization and TF-IDF vectorization and to see if there are most typo's in either the r/climate or the r/climateskeptics subreddit posts.

Some time was spent attempting to visualize how well the model predicted values, but I was not successful in doing this.  It would be nice to visualize how well models predicted values since the test data target values were available.

### Sources:

https://www.reddit.com/r/climate 

https://www.reddit.com/r/climateskeptics 

Nagpal, Anuja,
https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

Nicol, Will, https://www.digitaltrends.com/web/what-is-reddit/ 

Photo by Petter Rudwall on Unsplash, https://unsplash.com/search/photos/pollution?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText

Photo by Alto Crew on Unsplash,
https://unsplash.com/photos/Rv3ecImL4ak
