from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import time

def update_running_totals(new_values, running_count):
    if running_count is None:
        running_count = (0, 0.0)  # Initialize count and total sales.
    # Update the count and total sales respectively.
    return (running_count[0] + sum(x[0] for x in new_values), running_count[1] + sum(x[1] for x in new_values))

def format_output(rdd):
    # This converts the output to a readable string format
    return rdd.map(lambda record: f"Product ID: {record[0]}, Quantity Total: {record[1][0]}, Sales Total: {record[1][1]:.2f}")

def detect_anomalies(rdd):
    if not rdd.isEmpty():
        anomalies = rdd.filter(lambda record: record[1][0] > 10 or record[1][1] > 5000)  # Thresholds: quantity > 10 or sales > $5000
        formatted_anomalies = anomalies.map(lambda record: f"Anomaly Detected - Product ID: {record[0]}, Quantity: {record[1][0]}, Sales: {record[1][1]:.2f}")
        formatted_anomalies.saveAsTextFile(f"/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task5/anomalies_{int(time.time())}")

def main():
    sc = SparkContext(appName="RealTimeTransactionMonitoring")
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 1)  # 1 second window
    ssc.checkpoint("/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task5")  # Necessary for stateful transformations

    data_stream = ssc.socketTextStream("localhost", 9999)

    # Parse the incoming data
    transactions = data_stream.map(lambda line: line.split(","))
    keyed_transactions = transactions.map(lambda transaction: (transaction[0], (int(transaction[1]), float(transaction[3]))))

    # Update state
    running_totals = keyed_transactions.updateStateByKey(update_running_totals)

    # Format and write running totals to file
    formatted_totals = running_totals.transform(format_output)
    formatted_totals.foreachRDD(lambda rdd: rdd.saveAsTextFile(f"/workspaces/Project-2-Retail-Sales-Analytics-and-Real-Time-Demand-Forecasting_PRASHANTH_LAKKAKULA/output/task5/running_totals_{int(time.time())}"))

    # Detect anomalies
    running_totals.foreachRDD(detect_anomalies)

    ssc.start()
    ssc.awaitTermination()

if __name__ == '__main__':
    main()
