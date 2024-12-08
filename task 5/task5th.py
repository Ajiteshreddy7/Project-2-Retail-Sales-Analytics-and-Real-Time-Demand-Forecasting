import socket
import time
import random
import threading
from datetime import datetime
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col, sum as _sum

#######################################
# Step 1: Simulate Streaming Data
#######################################
def start_simulation(host="localhost", port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Simulating transactions on {host}:{port}, waiting for Spark connection...")

    conn, addr = server_socket.accept()
    print("Spark connected:", addr)

    products = ["P1", "P2", "P3", "P4"]
    try:
        while True:
            product_id = random.choice(products)
            quantity = random.randint(1, 20)
            price = round(random.uniform(5, 500), 2)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            transaction = f"{product_id},{quantity},{timestamp},{price}\n"
            conn.sendall(transaction.encode("utf-8"))
            time.sleep(1)
    except KeyboardInterrupt:
        server_socket.close()

# Start simulation in background thread
sim_thread = threading.Thread(target=start_simulation, args=("localhost", 9999), daemon=True)
sim_thread.start()

# Wait a moment for the simulation to start
time.sleep(5)

#######################################
# Step 2: Real-Time Processing with Spark Streaming
#######################################
spark = SparkSession.builder \
    .appName("RealTimeTransactionMonitoring") \
    .getOrCreate()

ssc = StreamingContext(spark.sparkContext, 5)

host = "localhost"
port = 9999
lines = ssc.socketTextStream(host, port)

def parse_transaction(line):
    fields = line.strip().split(",")
    if len(fields) == 4:
        return fields[0], int(fields[1]), fields[2], float(fields[3])
    else:
        return None

parsed = lines.map(parse_transaction).filter(lambda x: x is not None)

def process_rdd(rdd):
    if not rdd.isEmpty():
        df = spark.createDataFrame(rdd, ["product_id", "quantity", "timestamp", "price"])
        print("=== New Batch Received ===")
        df.show()

        # Running total sales
        running_totals = df.groupBy("product_id").agg(_sum("price").alias("total_sales"))
        print("=== Running Totals ===")
        running_totals.show()

        # Anomalies
        anomalies = df.filter((col("quantity") > 1) | (col("price") > 50))
        print("=== Anomalies ===")
        anomalies.show()

        # Write results to CSV (append mode)
        if not os.path.exists("running_totals_output"):
            os.makedirs("running_totals_output")
        if not os.path.exists("anomalies_output"):
            os.makedirs("anomalies_output")
        
        running_totals.write.mode("append").option("header", True).csv("running_totals_output")
        anomalies.write.mode("append").option("header", True).csv("anomalies_output")

parsed.foreachRDD(process_rdd)

print("Starting streaming...")
ssc.start()

# Let the streaming run for 120 seconds, then stop
ssc.awaitTerminationOrTimeout(120)
print("Stopping streaming...")
ssc.stop(stopSparkContext=False, stopGraceFully=True)
print("Streaming stopped.")

#######################################
# Step 3: Generate Logical Visualizations (Line & Scatter)
#######################################

# Read all running totals files and combine
rt_files = glob.glob("running_totals_output/*.csv")
if rt_files:
    dfs_rt = []
    for f in rt_files:
        df = pd.read_csv(f)
        dfs_rt.append(df)

    df_rt_all = pd.concat(dfs_rt, ignore_index=True)
    
    # Create a pseudo-time axis by using the row index as time steps.
    df_rt_all['time'] = df_rt_all.index  # pseudo-time

    # Create a line chart for total sales by product over pseudo-time
    plt.figure()
    for product_id in df_rt_all['product_id'].unique():
        product_data = df_rt_all[df_rt_all['product_id'] == product_id]
        plt.plot(product_data['time'], product_data['total_sales'], label=product_id)

    plt.title("Total Sales by Product Over Time")
    plt.xlabel("Time (batch index)")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("total_sales_over_time.png")
    plt.show()
else:
    print("No running totals data found for time-series visualization.")

# Read all anomalies files and combine
an_files = glob.glob("anomalies_output/*.csv")
if an_files:
    dfs_an = []
    for f in an_files:
        df = pd.read_csv(f)
        # Convert timestamp to datetime if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs_an.append(df)

    df_an_all = pd.concat(dfs_an, ignore_index=True)

    if not df_an_all.empty:
        # Sort by timestamp to plot in chronological order (if timestamps exist)
        if 'timestamp' in df_an_all.columns and df_an_all['timestamp'].notnull().any():
            df_an_all = df_an_all.sort_values('timestamp')

            # Scatter plot: anomalies price over time
            plt.figure()
            plt.scatter(df_an_all['timestamp'], df_an_all['price'], marker='x', color='red')
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.title("Anomalies Over Time (Price)")
            plt.tight_layout()
            plt.savefig("anomalies_over_time.png")
            plt.show()
        else:
            # If no valid timestamp, use pseudo-time based on index
            df_an_all['time'] = df_an_all.index
            plt.figure()
            plt.scatter(df_an_all['time'], df_an_all['price'], marker='x', color='red')
            plt.xlabel("Pseudo-Time (index)")
            plt.ylabel("Price")
            plt.title("Anomalies Over Time (Price)")
            plt.tight_layout()
            plt.savefig("anomalies_over_time.png")
            plt.show()
    else:
        print("No anomalies found for time-series visualization.")
else:
    print("No anomalies data found for time-series visualization.")
