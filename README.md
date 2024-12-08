# Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting

## Project Overview
This project provides hands-on experience in building a large-scale data processing pipeline for demand forecasting, customer sentiment analysis, and real-time analytics for a retail company. It integrates historical data analysis, real-time monitoring, and predictive modeling, creating a complex workflow.

## Objectives
- Process, analyze, and visualize historical retail data
- Forecast demand and analyze customer sentiment
- Implement real-time transaction monitoring using Spark Streaming

## Datasets
- **[Historical Sales Data: UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)** – Contains sales transaction data for an online retail store.
- **[Customer Reviews Data: Amazon Customer Reviews Dataset](https://registry.opendata.aws/amazon-reviews/)** – Includes detailed customer review data for sentiment analysis.

## Project Tasks

### Task 1: Data Loading, Cleansing, and Exploration
- **Data Ingestion**: Load the sales and review datasets into PySpark DataFrames. Explore and check for missing or inconsistent data.
- **Data Cleansing**: Handle missing values, drop duplicates, and parse dates for consistent data.

### Task 2: Sales Data Aggregation and Feature Engineering
- **Sales Aggregation**: Use PySpark SQL to calculate:
  - Total sales per product per month
  - Average revenue per customer segment
  - Seasonal patterns for top-selling products
- **Feature Engineering**: Derive features like customer lifetime value, product popularity score, and seasonal trends.

### Task 3: Demand Forecasting Model
- **Time Series Forecasting**: Using MLlib, forecast demand for popular products with ARIMA or lag-based feature models.
- **Model Pipeline**: Build a pipeline with feature assembly, data scaling, and model training. Evaluate with RMSE and MAE.
- **Hyperparameter Tuning**: Use CrossValidator for tuning model parameters.

### Task 4: Sentiment Analysis on Customer Reviews
- **Text Preprocessing**: Tokenize, remove stop words, and apply n-grams using PySpark NLP libraries.
- **Sentiment Model Training**: Train a Logistic Regression or Naive Bayes model on review data for sentiment classification.
- **Sentiment Scoring**: Calculate an aggregate sentiment score for each product.

### Task 5: Real-Time Transaction Monitoring using Spark Streaming
- **Streaming Data Simulation**: Simulate real-time transactions with Python sockets or Kafka, including fields like product ID, quantity, timestamp, and price.
- **Real-Time Processing**: Ingest transactions using Spark Streaming and calculate:
  - Running total sales for each product
  - Identify anomalies with unusual quantities or prices
- **Real-Time Dashboard**: Write results to a dashboard application (e.g., Plotly Dash or database).

### Advanced Task (Optional): Customer Segmentation and Personalized Recommendations
- **Customer Segmentation**: Use clustering (e.g., KMeans) to segment customers by purchase behavior.
- **Product Recommendations**: Implement collaborative filtering (ALS) to recommend products based on customer segments and sentiment scores.

## Deliverables
- **Codebase**: Implement each task in well-structured PySpark scripts with documentation.
- **Documentation**: Detailed explanations of each approach, challenges, and findings.
- **Final Report**: A presentation summarizing insights and recommendations.
- **Real-Time Dashboard**: Visualization of real-time transactions and streaming analysis.

## Summary
This project integrates PySpark DataFrames, Spark SQL, MLlib, and Spark Streaming to tackle end-to-end data engineering and machine learning workflows in a retail setting. Students will reinforce skills in data transformation, modeling, and real-time analytics, building a comprehensive foundation in PySpark and Big Data processing.


---

## **Commit and Push Your Work**
Once you have completed the tasks and generated the output files:
1. Add the changes to your GitHub repository:
```bash
git add .
git commit -m "Completed Project-2"
git push origin main
```

2. Ensure that all code and output files (CSV files) are pushed to your repository.

---
