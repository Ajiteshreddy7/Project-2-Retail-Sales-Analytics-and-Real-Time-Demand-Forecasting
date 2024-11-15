from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, month, to_date

# Initialize a SparkSession with the legacy time parser policy
spark = SparkSession.builder \
    .appName("Sales Data Aggregation and Feature Engineering") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Define file paths
online_retail_data_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/input/Online Retail.csv"

# Define output paths
output_path = "/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task2"  # Base path for all output files

# Load the dataset from CSV file
online_retail_df = spark.read.csv(online_retail_data_path, header=True, inferSchema=True)

# Convert date column to proper date format (adjust the date format string as necessary)
online_retail_df = online_retail_df.withColumn("InvoiceDate", to_date(col("InvoiceDate"), "MM/dd/yyyy"))

# Sales Aggregation: Total sales per product per month
total_sales_per_product_per_month = online_retail_df.groupBy("StockCode", month("InvoiceDate").alias("month")).agg(sum("Quantity").alias("total_sales"))
total_sales_per_product_per_month.write.csv(output_path + "/total_sales_per_product_per_month.csv", header=True, mode="overwrite")

# Sales Aggregation: Average revenue per customer
average_revenue_per_customer = online_retail_df.groupBy("CustomerID").agg(avg("UnitPrice").alias("average_revenue"))
average_revenue_per_customer.write.csv(output_path + "/average_revenue_per_customer.csv", header=True, mode="overwrite")

# Sales Aggregation: Seasonal patterns for top-selling products
top_selling_products = online_retail_df.groupBy("StockCode").agg(sum("Quantity").alias("total_sales")).orderBy(col("total_sales").desc())
top_selling_products.write.csv(output_path + "/top_selling_products.csv", header=True, mode="overwrite")

# Feature Engineering: Customer lifetime value
customer_lifetime_value = online_retail_df.groupBy("CustomerID").agg(sum("Quantity").alias("lifetime_value"))
customer_lifetime_value.write.csv(output_path + "/customer_lifetime_value.csv", header=True, mode="overwrite")

# Feature Engineering: Product popularity score
product_popularity_score = online_retail_df.groupBy("StockCode").count().withColumnRenamed("count", "popularity_score")
product_popularity_score.write.csv(output_path + "/product_popularity_score.csv", header=True, mode="overwrite")

# Feature Engineering: Seasonal trends
seasonal_trends = online_retail_df.groupBy("StockCode", month("InvoiceDate").alias("month")).agg(sum("Quantity").alias("monthly_sales"))
seasonal_trends.write.csv(output_path + "/seasonal_trends.csv", header=True, mode="overwrite")

# Stop the Spark session
spark.stop()
