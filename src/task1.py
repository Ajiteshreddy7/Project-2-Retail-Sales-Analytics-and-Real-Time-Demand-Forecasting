from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date

# Initialize a SparkSession
spark = SparkSession.builder \
    .appName("Retail Sales and Demand Forecasting") \
    .getOrCreate()

# Define file paths
sales_data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/input/train.csv"
reviews_data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/input/Online Retail.csv"

# Define output paths
output_sales_data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/cleaned_sales_data.csv"
output_reviews_data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/cleaned_reviews_data.csv"

# Load the datasets from CSV files
sales_df = spark.read.csv(sales_data_path, header=True, inferSchema=True)
reviews_df = spark.read.csv(reviews_data_path, header=True, inferSchema=True)

# Print the schema to understand data types
print("Sales Data Schema:")
sales_df.printSchema()
print("Reviews Data Schema:")
reviews_df.printSchema()

# Count the number of missing values in each column
print("Missing Values in Sales Data:")
for column in sales_df.columns:
    missing_count = sales_df.filter(sales_df[column].isNull()).count()
    if missing_count > 0:
        print(f"{column}: {missing_count}")

print("Missing Values in Reviews Data:")
for column in reviews_df.columns:
    missing_count = reviews_df.filter(reviews_df[column].isNull()).count()
    if missing_count > 0:
        print(f"{column}: {missing_count}")

# Data Cleansing
# Drop duplicates
sales_df = sales_df.dropDuplicates()
reviews_df = reviews_df.dropDuplicates()

# Handle missing values and parse dates if 'date' column is present in sales data
if 'date' in sales_df.columns:
    sales_df = sales_df.na.drop(subset=["date"])  # Drops rows where 'date' is null
    sales_df = sales_df.withColumn("date", to_date(sales_df["date"], "yyyy-MM-dd"))  # Convert date format
else:
    print("No 'date' column found in sales data")

# Drop rows with missing review data
reviews_df = reviews_df.na.drop()

# Output the cleaned data to CSV files
sales_df.write.csv(output_sales_data_path, header=True, mode="overwrite")
reviews_df.write.csv(output_reviews_data_path, header=True, mode="overwrite")

# Print confirmation of saved files
print(f"Cleaned sales data saved to {output_sales_data_path}")
print(f"Cleaned reviews data saved to {output_reviews_data_path}")

# Stop the Spark session
spark.stop()
