from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, year, month, sum
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Retail Sales Demand Forecasting") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Load Data
data = spark.read.csv('/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/input/Online Retail.csv', header=True, inferSchema=True)

# Data Preprocessing
data = data.withColumn("InvoiceDate", to_date(col("InvoiceDate"), "MM/d/yyyy H:mm"))
data = data.dropna(how='any')

# Feature Engineering
data = data.withColumn('Year', year(col('InvoiceDate')))
data = data.withColumn('Month', month(col('InvoiceDate')))
data = data.groupBy('Year', 'Month', 'StockCode').agg(sum('Quantity').alias('TotalQuantity'))

# Assemble Features
assembler = VectorAssembler(inputCols=['Year', 'Month'], outputCol='features')

# Feature Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Model Building
lr = LinearRegression(featuresCol='scaledFeatures', labelCol='TotalQuantity')

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Model Evaluation and Hyperparameter Tuning
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = RegressionEvaluator(labelCol="TotalQuantity")
crossval = CrossValidator(estimator=pipeline, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator,
                          numFolds=3)

# Fit Model
cvModel = crossval.fit(data)

# Predict and Evaluate
predictions = cvModel.transform(data)
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

# Writing RMSE to a file
with open('/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task3_output/rmse.txt', 'w') as f:
    f.write(f"Root Mean Square Error (RMSE) on test data = {rmse}\n")

# Writing MAE to a file
with open('/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task3_output/mae.txt', 'w') as f:
    f.write(f"Mean Absolute Error (MAE) on test data = {mae}\n")

# Stop Spark Session
spark.stop()
