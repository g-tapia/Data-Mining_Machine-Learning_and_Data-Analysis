# Data-Mining (Machine Learning)

Note: Rmd files contain the code, html files contain the output, the code, and the notes. There are 9 labs I worked on. I also used multiple libraries (tensorflow is one of them).

Feel free to read an article I wrote above. I used overleaf to format it. Though, it did take me a long time to write it since I had to do extensive research, make the tables myself, and get familiar with overleaf. 

I Wasn't able to write more articles due to the fact that I was going the extra mile on my homework and wasting more time than I needed to. Still, I am happy with what I wrote. I did way more work than the graduate students in my class. 

I wrote the article voluntarily. I will post the other article in the next few days. 


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


# Advanced Decision Trees (US Census Dataset)

### Theoretical Understanding
In this homework, I strengthened my theoretical knowledge by answering questions from Tan's book focusing on different aspects of decision tree algorithms and multiclass classification problems. I practiced calculating sensitivity, specificity, and precision, further enhancing my understanding of these evaluation metrics in machine learning.

### Data Cleaning and Preprocessing
The assignment involved handling the 1994 US Census datasets, containing several incomplete entries. The initial task required identifying and removing such instances, transforming the dataset into a more reliable form for data mining. This process significantly improved my skills in data preparation, a crucial step before performing any data mining or machine learning tasks.

### Advanced Decision Tree Classification
The core part of the assignment was the practical application of advanced decision tree classification using R's `rpart` function on the cleaned dataset. The goal was to predict whether a person's income exceeds $50K per year, based on various features such as age, work class, education, and others. This section helped me gain a deeper understanding of decision trees and their applicability in real-world predictive modeling tasks.

### Model Analysis
After building the decision tree model, I analyzed the model to identify key predictors and understand the initial split in the tree. This analysis provided valuable insights into how decision trees work and how they select features for splits. 

### Model Evaluation
I evaluated the model's performance using the test dataset and calculated crucial metrics such as balanced accuracy, balanced error rate, sensitivity, specificity, and the Area Under the ROC curve. This step underscored the importance of these evaluation metrics in understanding a model's effectiveness and reliability.

### Addressing Class Imbalance
The training dataset exhibited a class imbalance problem. I addressed this issue by employing undersampling of the majority class, training a new model, and then comparing its performance with the original model. This step facilitated a better understanding of the impact of class imbalance on a model's performance and how to mitigate it.

### Model Comparison
Finally, I compared the balanced accuracy, sensitivity, specificity, positive predictive value, and AUC of the two models. This comparison helped me comprehend the effects of class imbalance, and its correction, on the overall performance of predictive models.

Overall, this assignment solidified my understanding of advanced decision trees and their practical application in predicting income based on various demographic and employment-related attributes.

# Association Rule Mining (Bakery dataset)

### Theoretical Understanding
In this assignment, theoretical knowledge was further strengthened by answering questions from Tan's book on Association Analysis and Zaki's chapter on Frequent Pattern Mining. The questions facilitated a deeper understanding of association rules, frequent pattern mining, and the application of these concepts in data mining.

### Data Transformation
The assignment used the 'Extended Bakery' dataset, a real-world dataset representing transactions from a chain of bakery shops. The dataset came in a sparse vector format, with each transaction containing a list of purchased items represented by their product ID code. The first task was to transform this dataset into a more interpretable canonical representation using the mapping provided in the products.csv file. This required translating the product ID codes into actual product names, thereby creating an easy-to-read file for each series of transactions (tr-5k, tr-20k, etc.). This task enhanced my data manipulation skills and provided experience in creating user-friendly datasets.

### Association Rule Mining
The main part of the assignment involved finding association rules that associate the presence of one set of items with another set, using the arules package in R. This was done for each series of transactions, using varying levels of minimum support and confidence. The output included the frequent itemsets with their support and association rules with the antecedent, consequent, support, and confidence. This exercise gave me valuable experience in using association analysis to mine for frequent itemsets and strong association rules in a dataset.

### Comparison and Interpretation
The last part of the assignment required comparing the rules obtained for different subsets of transactions and analyzing how the number of transactions impacts the results. Additionally, for the 75,000 transactions dataset, I had to identify the most and least frequently purchased item or itemset. These tasks allowed me to reflect on the influence of data size on results and to draw meaningful insights from the data.

Through these practical tasks, this assignment solidified my understanding of association rule mining and its application in analyzing transaction data. I gained experience in transforming data, using the arules package in R for mining association rules, and interpreting these rules for actionable insights.


# Implementing Perceptron in R

### Understanding Perceptron
In this assignment, I deepened my understanding of the Perceptron, a fundamental machine learning algorithm for binary classification problems. I studied how the algorithm uses a linear predictor function based on a set of weights that get updated as the model learns from the training data.

### Implementation of Perceptron in R
The heart of the assignment was to manually implement the Perceptron algorithm in R. I developed a function that initializes random weights, iterates over the training data, and updates the weights whenever it encounters a misclassified instance. The implementation also included a parameter for the learning rate, which modulates the magnitude of the weight updates. This exercise was fundamental to my understanding of how Perceptron works and its implementation details.

### Setting Learning Rate and Steps
Choosing the right learning rate was crucial in this assignment as it directly influences the speed and quality of learning. Similarly, I had to set an appropriate number of steps or iterations over the dataset for the Perceptron to converge to an optimal solution. This task honed my skills in hyperparameter tuning and understanding its impact on the model's performance.

### Model Evaluation
After building the Perceptron model, I used it to make predictions and evaluated its performance using a set of appropriate metrics. These included accuracy, precision, recall, and F1-score. The evaluation helped me comprehend the real-world performance of the Perceptron model and the implications of the chosen hyperparameters.

### Visualization
To better interpret the model, I created visualizations to illustrate the Perceptron's decision boundary in the feature space. This provided valuable insights into the model's decision-making process and the impact of the weight updates throughout the learning phase.

In conclusion, through this assignment, I gained a solid understanding of the Perceptron algorithm, its implementation details, and performance tuning. I now appreciate the importance of setting appropriate learning rates and iteration steps and how these affect the model's performance and learning quality.

# Neural Network Implementation and Optimization

### Theoretical Understanding
This assignment concentrated on theoretical and practical facets of neural networks. The theoretical part offered questions about neural network architecture, error computation, backpropagation, and weight adjustments. It also delved into the bias-variance trade-off concept. These exercises helped deepen my understanding of neural networks and their mathematical basis.

### Data Preprocessing
The assignment required handling the 'wifi_localization.csv' dataset, encompassing signals from seven WiFi access points for predicting a user's location among four rooms. This dataset was divided into a training set (80%) and a test set (20%), forming the basis for the modeling tasks.

### Decision Tree Baseline Model
To set a performance benchmark, a decision tree was first trained on the dataset and evaluated on the test data. The performance of the decision tree model was assessed via a confusion matrix, looking at overall accuracy, sensitivity, specificity, and positive predictive value for each class. This part of the assignment emphasized my understanding of decision tree models and how to interpret confusion matrices.

### Single-Neuron Hidden Layer Neural Network
Following the decision tree model, a basic neural network with a single neuron in the hidden layer was implemented. The hidden layer employed the ReLU activation function, and the output layer used softmax. After training the model for 100 epochs, its performance was gauged using the test data. The model's effectiveness was primarily evaluated based on loss and accuracy metrics.

### Optimized Neural Network with Variable Neurons in Hidden Layer
In the next step, the objective was to enhance the model's accuracy by expanding the number of neurons in the hidden layer. This step involved iterative training and assessment of the model until the accuracy reached high levels. This task illustrated the importance of the number of neurons in a hidden layer for a neural network's performance.

### Model Evaluation and Comparison
The final task was to evaluate the enhanced neural network model, presenting its performance in terms of similar metrics used for the decision tree model. The performance of both models was then compared, and a conclusion was made regarding the preferred model for deployment. This comparison illuminated the respective strengths and weaknesses of decision trees and neural networks when applied to a real-world dataset.

In summary, this assignment bolstered my understanding of neural networks, their implementation in R, and their performance comparison with decision tree models.



# Advanced Clustering Techniques: K-Means and DBSCAN

### Theory Component
In this assignment, I answered questions from Tan's book, which concentrated on the fundamentals and algorithms of cluster analysis. The questions were primarily focused on distance and similarity measures, hierarchical and density-based clustering, and K-means. It helped reinforce my understanding of the clustering concepts and their real-world applications.

### Data Preprocessing
I worked with the HARTIGAN dataset (file19.txt), which is a multivariate mammal dataset with 66 rows and 9 columns. I performed data cleaning operations such as removal of specific attributes, multiple spaces, standardization, and conversion of the delimiter to a comma. This part of the assignment emphasized the importance of clean and well-prepared data before applying any clustering algorithm.

### K-Means Clustering Implementation and Analysis
I conducted K-means clustering on the HARTIGAN dataset. The process included determining the optimal number of clusters using the WSS or Silhouette graph, running K-means clustering, and analyzing the clusters using various metrics such as SSE. A detailed examination of each cluster helped understand how the mammals were grouped together, leading to fascinating insights about the animal kingdom and the efficiency of K-means clustering.

### DBSCAN Clustering
This assignment also required applying the DBSCAN clustering algorithm on the dataset s1.csv, which contains 5,000 observations of two dimensions. The tasks included plotting the dataset, deciding on the optimal number of clusters using the K-means algorithm for comparison, performing DBSCAN clustering, and determining appropriate parameters for MinPts and epsilon. Analyzing the DBSCAN results in comparison to K-means clustering shed light on the advantages and disadvantages of both clustering techniques.

Overall, this assignment provided an in-depth understanding of advanced clustering techniques and their practical application on real-world datasets. It further reinforced my comprehension of data preparation and clustering-based data analysis.
