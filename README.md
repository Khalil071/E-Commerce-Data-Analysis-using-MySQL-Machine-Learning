E-Commerce Data Analysis using MySQL & Machine Learning

Overview

This project focuses on analyzing e-commerce data using the wp-ecommerce dataset. The data is stored and managed in MySQL, and machine learning techniques, specifically the Decision Tree Classifier, are applied to classify customer behaviors. Additionally, m2cgen is used to convert the trained model into code for deployment.

Features

Perform exploratory data analysis (EDA) on the wp-ecommerce dataset

Use MySQL to store and query e-commerce data efficiently

Train a Decision Tree Classifier to predict customer behavior

Convert the trained model into portable code using m2cgen

Visualize key insights from the data

Technologies Used

MySQL

Python

Pandas & NumPy

Scikit-learn (Decision Tree Classifier)

m2cgen (Model to Code Generator)

Matplotlib & Seaborn (Data Visualization)

Installation

Clone the repository:

git clone https://github.com/yourusername/ecommerce-analysis.git
cd ecommerce-analysis

Install dependencies:

pip install -r requirements.txt

Set up MySQL database:

Install MySQL and create a database

Import the wp-ecommerce dataset into MySQL

CREATE DATABASE ecommerce;
USE ecommerce;
-- Import dataset using MySQL Workbench or CLI

Data Analysis & Model Training

Load data from MySQL:

import mysql.connector
import pandas as pd

conn = mysql.connector.connect(host='localhost', user='root', password='yourpassword', database='ecommerce')
query = "SELECT * FROM wp_ecommerce"
df = pd.read_sql(query, conn)
conn.close()

Train Decision Tree Classifier:

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df.drop(columns=['target_column'])  # Replace with relevant features
y = df['target_column']  # Replace with the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

Convert model to code using m2cgen:

import m2cgen as m2c

generated_code = m2c.export_to_python(model)
with open("model.py", "w") as f:
    f.write(generated_code)

Future Improvements

Optimize SQL queries for better data retrieval efficiency

Fine-tune the Decision Tree Classifier for improved accuracy

Extend the model with ensemble methods (Random Forest, XGBoost)

Deploy the model as an API for real-time predictions

License

This project is licensed under the MIT License.
