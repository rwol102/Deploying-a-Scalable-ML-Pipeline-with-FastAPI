# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Algorithm: Random Forest Classifier
Task: Binary classification — predict whether a person earns >50K or <=50K annually
Frameworks: scikit-learn, pandas, FastAPI (deployment), pickle (serialization)
Data Source: UCI Census Income Dataset

## Intended Use
This model is intended for educational demonstration in a MLOps deployment pipeline. It 
is designed to:
Showcase the integration of model training, evaluation, and serving
Demonstrate fairness checks using performance slices
Be deployed in a production-like REST API using FastAPI

## Training Data
Source: UCI Adult Census Income dataset
File used: data/census.csv
Split: 80% training, 20% test using train_test_split
Label: salary
Categorical Features:
    workclass, education, marital-status, occupation, relationship, race, sex, native-country
Preprocessing script: ml/data.py → process_data()

## Evaluation Data
For this particular model 20% was used for testing the model while 80% was initially used for the training of the model.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
F1 score: 0.6863
Metric socre
Precision: 0.7419
Recall: 0.6384


## Ethical Considerations
The model may reflect historical and societal bias in the census data, especially across  
race, gender, or nationality.
There is risk of disparate performance across demographic groups — analyzed partially using 
slice evaluation.
Misclassification could cause real-world harm if used in sensitive applications (e.g., loan 
approval, hiring).

## Caveats and Recommendations
Not production-ready: Only intended for learning and demonstration purposes.
Categorical feature encoding is tightly coupled with the training data — new categories at 
inference time may be ignored.
Model fairness evaluation is limited to slice metrics and does not cover causal or 
statistical parity checks.
Further evaluation using tools like Aequitas, Fairlearn, or SHAP is recommended before any 
real-world deployment.