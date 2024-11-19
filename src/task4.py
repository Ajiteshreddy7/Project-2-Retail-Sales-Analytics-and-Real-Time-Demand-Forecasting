from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# Define a UDF to concatenate arrays
@udf(ArrayType(StringType()))
def concat_arrays(arr1, arr2):
    if arr1 is None:
        arr1 = []
    if arr2 is None:
        arr2 = []
    return arr1 + arr2

# Load data
data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/input/train.csv"  # Update the path to your CSV file
data = spark.read.csv(data_path, header=True, inferSchema=True).select("sentence", "main_image_url")

# Rename column for clarity
data = data.withColumnRenamed("main_image_url", "product_id")

# Text Preprocessing and Feature Engineering Pipeline
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")
hashingTF = HashingTF(inputCol="features_combined", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Model
lr = LogisticRegression(featuresCol='features', labelCol='label')

# Define the pipeline
pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    ngram,
    udf(concat_arrays(col("filtered"), col("ngrams")).alias("features_combined")),
    hashingTF,
    idf,
    lr
])

# Prepare labels (Assuming labels are in the data for simplicity, adjust according to your needs)
data = data.withColumn("label", when(col("sentence").isNull(), 0).otherwise(1))

# Split data
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train the model
model = pipeline.fit(trainingData)

# Predictions
predictions = model.transform(testData)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

# Calculate average sentiment per product
product_sentiment = predictions.groupBy("product_id").agg(avg("prediction").alias("avg_sentiment"))
product_sentiment.write.csv("/path_to_output/product_sentiments.csv")

# Save evaluation results to a file
with open("/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task4_output.txt", "w") as file:
    file.write(f"Area Under ROC: {auc}\n")

# Stop Spark session
spark.stop()
