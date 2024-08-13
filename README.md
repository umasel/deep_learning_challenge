# Deep Learning Challenge : Charity Funding Predictor

## Solution

Solution to predict whether or not applicants for Charity Funding will be successful based the process of Deep Learning.

## Background
The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Instructions
### Step 1: Preprocess the data
Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2

Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
           - What variable(s) are considered the target(s) for your model?
           - What variable(s) are considered the feature(s) for your model?
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables

### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the jupter notebook where you’ve already performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every 5 epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file, and name it 
AlphabetSoupCharity.h5.

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

- Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
        - Dropping more or fewer columns.
        - Creating more bins for rare occurrences in columns.
        - Increasing or decreasing the number of values for each bin.
- Adding more neurons to a hidden layer.
- Adding more hidden layers.
- Using different activation functions for the hidden layers.
- Adding or reducing the number of epochs to the training regimen.

NOTE: You will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your jupyter notebook.

1. Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
2. Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
3. Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.

# Final Analysis and Report on the Neural Network Model

Below is my final report and analysis of the Neural Network Model, along with answers to the questions posed in the assignment:

## Overview of the Analysis:

- Purpose: The objective of this analysis was to develop an algorithm for Alphabet Soup to predict the likelihood of success for applicants seeking funding. The model is a binary classifier designed to predict with a reasonably high level of accuracy whether the funding will be successful or not.

## Results:

### Data Preprocessing:

- Target Variable: The target variable for the model was identified as the column IS_SUCCESSFUL.
- Feature Variables: The following columns were used as features for the model:
  - NAME
  - APPLICATION_TYPE
  - AFFILIATION
  - CLASSIFICATION
  - USE_CASE
  - ORGANIZATION
  - STATUS
  - INCOME_AMT
  - SPECIAL_CONSIDERATIONS
  - ASK_AMT
- Variables to Remove: The EIN column was removed, as it serves only as an identifier for the applicant organization and does not affect the model's behavior.

## Compiling, Training, and Evaluating the Model:

- Model Architecture: The optimized model used 3 hidden layers with multiple neurons, which increased the accuracy from under 75% to 79%. The initial model had only 2 layers. While the number of epochs remained constant, adding a third layer improved the model's accuracy.
- Target Performance: Yes, by optimizing the model, the accuracy increased from 72% to slightly over 79%.
- Steps to Increase Performance:
             - Instead of dropping both the EIN and NAME columns, only the EIN column was dropped. However, only names that appeared more than 5 times were considered.
             - A third activation layer was added to the model in the following order to boost accuracy to over 75%:
                      - 1st Layer: ReLU
                      - 2nd Layer: Tanh
                      - 3rd Layer: Sigmoid
             - It was observed that using Tanh for the 2nd layer and Sigmoid for the 3rd layer boosted performance to over                   79%.

## Summary and Recommendation:

Overall, optimizing the model increased accuracy to over 79%. This means that the model correctly classified the test data 79% of the time. Applicants have nearly an 80% chance of being successful if they meet the following criteria:

- The applicant's name appears more than 5 times (indicating they have applied more than 5 times).
- The application type is one of the following: T3, T4, T5, T6, or T19.
- The application has one of the following classification values: C1000, C1200, C2000, C2100, or C3000.

# Alternative Method:

While this model performed well and provided substantial accuracy, an alternative approach to consider is the Random Forest model, which is also well-suited for classification problems. Using the Random Forest model, we can achieve an accuracy close to 78%.






