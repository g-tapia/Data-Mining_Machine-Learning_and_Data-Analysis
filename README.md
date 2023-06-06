# Data-Mining (Machine Learning)

Note: Rmd files contain the code, html files contain the output of the code, the code, and the notes. There are 9 labs I worked on.


# Data Analysis of Advertising Dataset

In this analysis, I explored and analyzed an advertising dataset that contained sales data across 200 different markets and advertising budgets allocated for three media channels - TV, Radio, and Newspaper.

1. **Data Manipulation and Visualization:** I started the analysis by importing the dataset into a dataframe and performing initial data manipulations, such as printing specific rows and columns. To visualize the relationship between sales and each advertising channel, I created scatter plots for sales versus TV, radio, and newspaper advertising budgets.

2. **Exploratory Data Analysis:** By studying these scatter plots, I gained insights into the 'noisiness' of each plot, which indicated the clarity of any observable trend in the data. The goal was to understand which medium had a less clear relationship with sales.

3. **Correlation Analysis:** I then explored the correlations among the data using the `corPlot()` function from the 'psych' library in RStudio. I created a heatmap to visualize these correlations, which helped me understand which mode of advertising had the highest positive correlation with sales and which had the most negative correlation.

4. **Data Distribution Analysis:** I created box plots for each advertising medium to understand their distribution. Box plots are a great way to visualize the spread and skewness of data, along with any potential outliers.

5. **Feature Engineering:** To further understand the advertising budget allocation, I created a new 'total' feature that represented the total money spent on advertising (TV, Radio, and Newspaper) for each market. I then filtered the data to display only those markets where the total advertising budget exceeded $400,000.

6. **Data Investigation:** Lastly, I identified the market with the maximum sales and compared its total advertising budget with the markets filtered in the previous step. This allowed me to make inferences about the relationship between total advertising budget and sales.

By analyzing the advertising dataset, I got hands-on experience manipulating, visualizing, and understanding data. I learned how to analyze relationships between variables using scatter plots and heatmaps, understand data distributions with boxplots, and perform feature engineering and data filtering to gain specific insights. This experience will be beneficial in any data analysis or data science role.

# Multivariate Linear Regression

In this assignment, I engaged in understanding and applying concepts related to Multivariate Linear Regression. 

Here is a picture of my notes, explaining linear regression, in case you need a refresher, or curious to have a basic understanding.

<img width="595" alt="image" src="https://github.com/g-tapia/Data-Mining---Artificial-Intelligence/assets/78235399/5484667f-b150-4929-b1ee-a26a5acd66e1">

1. **Theoretical Understanding:** The homework started with exercises from the "An Introduction to Statistical Learning with Applications in R (ISLR)" book. This helped me understand the Least Squares Line in linear regression and strengthened my theoretical knowledge.

2. **Hands-on Experience with Regression Modeling:**

   - **Dataset Manipulation:** I had to manipulate the 'Auto' dataset from the ISLR package in R. The dataset was split into training (95%) and testing (5%) sets, allowing me to practice model development and subsequent validation.

   - **Building and Assessing a Model:** I built a linear regression model using all features except 'name' to predict the 'mpg' (miles per gallon) target variable. After building the model, I examined its performance using R-squared, Adjusted R-squared, RSE (Residual Standard Error), and RMSE (Root Mean Square Error) metrics. 

   - **Residual Analysis:** I plotted the residuals of the model and created a histogram to analyze their distribution. This was an essential step in verifying the assumption of normally distributed residuals in linear regression.

   - **Feature Selection:** I learned to refine the regression model by identifying and selecting significant predictors. I narrowed down the features to the three most significant predictors and created a new model. I then evaluated this new model using the same performance metrics and residual analysis.

3. **Model Validation and Prediction Accuracy:** The last part of the assignment involved testing the model's performance on unseen data (the test set). I used the `predict()` function to generate predictions, then assessed their accuracy by creating confidence and prediction intervals and checking how many of the true values fell within these intervals.

4. **Comparison and Interpretation:** Finally, I compared the prediction accuracies from the confidence interval and the prediction interval methods, identified which method resulted in more accurate predictions, and discussed the reasons for this.

This assignment significantly enhanced my practical understanding of Multivariate Linear Regression, including model building, assessment, and validation processes.

# Decision Tree Classification (Hotels Dataset)

### Theoretical Understanding
The homework kicked off with questions from Tan's Data Mining book, which provided a solid theoretical understanding of various aspects of data mining, including principles and techniques.


### Data Exploration and Manipulation
The assignment introduced me to a real-world 'Hotels Bookings' dataset, comprised of 32 dimensions and 119,390 observations. The data exploration tasks required handling this dataset using R programming, interpreting the dataset attributes, and analyzing different aspects of the data.

### Exploratory Data Analysis (EDA)
In the Exploratory Data Analysis section, I learned to count observations for different hotel types, analyze the distribution of class labels, and identify the customer type with the most reservations. I also discovered how to determine the maximum and minimum number of parking spaces required by customers and compare the preferred room type with the assigned room type at check-in. Furthermore, I was able to identify the top 10 countries of origin for bookings for each hotel type. Through these tasks, I learned how to manipulate data, use various R commands, create visualizations, and draw meaningful conclusions from the data.

### Data Cleaning
Before creating my decision tree model, I first created a function to apply to all the dimensions in the dataset. This function helped me determined which attributes were missing values by percentanges.

Next, I displayed these percentages on a table to perform an analysis.

<img width="482" alt="image" src="https://github.com/g-tapia/Data-Mining---Artificial-Intelligence/assets/78235399/fea0dfe9-4007-4ff3-b181-1a9685197f34">

Upon reading a few articles on medium, I discovered that it is generally better to drop the columns with 70% or more missing data, so I dropped company. 

For the rest of the attributes, I took the mean of each column and filled them in with the mean. Although, for agent, I just filled it in with the mode.

### Decision Tree Modeling
In the next part, I gained practical experience in creating decision tree models using the rpart package in R. I used the same 'Hotels Bookings' dataset to predict whether a booking will be canceled or not, splitting the data into training and testing sets.

### Model Building, Assessment, and Validation
Building the decision tree model involved thoughtful selection of predictor variables. After model creation, I had to plot the decision tree, identify the important variables, fit the model on the test dataset, and calculate model metrics including Accuracy, Error, Balanced Accuracy, Specificity, Sensitivity, and Precision. This practice allowed me to grasp the model development and subsequent validation processes better.

### Interpretation and Application
The assignment concluded with an interpretation of the model metrics and an application of the model to unseen data (the test set). This gave me practical experience in assessing the quality of a model, predicting outcomes, and validating those predictions.

Overall, this assignment significantly enriched my practical understanding of data mining techniques, particularly exploratory data analysis and decision tree modeling. It was an engaging exercise that bridged the gap between theoretical concepts and their practical applications.
