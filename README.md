# Data-Mining---Artificial-Intelligence

Note: Rmd files contain the code, html files contain the output of the code and the notes. There are 9 labs I worked on, so far I put two, will upload the rest throughout this week.

# Data Analysis of Advertising Dataset

In this analysis, I explored and analyzed an advertising dataset that contained sales data across 200 different markets and advertising budgets allocated for three media channels - TV, Radio, and Newspaper.

## What I Learned 

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

## What I learned

1. **Theoretical Understanding:** The homework started with exercises from the "An Introduction to Statistical Learning with Applications in R (ISLR)" book. This helped me understand the Least Squares Line in linear regression and strengthened my theoretical knowledge.

2. **Hands-on Experience with Regression Modeling:**

   - **Dataset Manipulation:** I had to manipulate the 'Auto' dataset from the ISLR package in R. The dataset was split into training (95%) and testing (5%) sets, allowing me to practice model development and subsequent validation.

   - **Building and Assessing a Model:** I built a linear regression model using all features except 'name' to predict the 'mpg' (miles per gallon) target variable. After building the model, I examined its performance using R-squared, Adjusted R-squared, RSE (Residual Standard Error), and RMSE (Root Mean Square Error) metrics. 

   - **Residual Analysis:** I plotted the residuals of the model and created a histogram to analyze their distribution. This was an essential step in verifying the assumption of normally distributed residuals in linear regression.

   - **Feature Selection:** I learned to refine the regression model by identifying and selecting significant predictors. I narrowed down the features to the three most significant predictors and created a new model. I then evaluated this new model using the same performance metrics and residual analysis.

3. **Model Validation and Prediction Accuracy:** The last part of the assignment involved testing the model's performance on unseen data (the test set). I used the `predict()` function to generate predictions, then assessed their accuracy by creating confidence and prediction intervals and checking how many of the true values fell within these intervals.

4. **Comparison and Interpretation:** Finally, I compared the prediction accuracies from the confidence interval and the prediction interval methods, identified which method resulted in more accurate predictions, and discussed the reasons for this.

This assignment significantly enhanced my practical understanding of Multivariate Linear Regression, including model building, assessment, and validation processes.
