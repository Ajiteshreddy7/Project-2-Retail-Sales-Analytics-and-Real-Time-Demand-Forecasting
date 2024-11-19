from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# Define a UDF to concatenate arrays
concat_arrays_udf = udf(lambda x, y: x + y, ArrayType(StringType()))

# Load data
data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/input/train.csv"  # Update the path to your CSV file
data = spark.read.csv(data_path, header=True, inferSchema=True).select("sentence", "main_image_url")

# Rename column for clarity
data = data.withColumnRenamed("main_image_url", "product_id")

# Text Preprocessing
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")

# Transform the data through the initial stages
pipeline = Pipeline(stages=[tokenizer, remover, ngram])
transformed_data = pipeline.fit(data).transform(data)

# Combine the NGram and filtered columns into a single column for feature engineering
transformed_data = transformed_data.withColumn("rawFeatures", concat_arrays_udf(col("filtered"), col("ngrams")))

# Feature Engineering using HashingTF and IDF
hashingTF = HashingTF(inputCol="rawFeatures", outputCol="features", numFeatures=10000)
featurized_data = hashingTF.transform(transformed_data)

idf = IDF(inputCol="features", outputCol="finalFeatures")
final_data = idf.fit(featurized_data).transform(featurized_data)

# Prepare labels (assuming 'sentence' is not null when labeled)
final_data = final_data.withColumn("label", col("sentence").isNotNull().cast("int"))

# Split data into training and test sets
(trainingData, testData) = final_data.randomSplit([0.7, 0.3])

# Logistic Regression Model
lr = LogisticRegression(featuresCol='finalFeatures', labelCol='label')
final_model = lr.fit(trainingData)

# Predictions
predictions = final_model.transform(testData)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

# Calculate average sentiment per product
product_sentiment = predictions.groupBy("product_id").agg({"prediction": "avg"}).withColumnRenamed("avg(prediction)", "avg_sentiment")
product_sentiment.write.csv("/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task4/product_sentiments.csv")

# Save evaluation results to a file
with open("/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task4/task4_output.txt", "w") as file:
    file.write(f"Area Under ROC: {auc}\n")

# Stop Spark session
spark.stop()
