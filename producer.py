from kafka import KafkaProducer
import csv
import time
from concurrent.futures import ThreadPoolExecutor

def produce_to_kafka(topic, file_path):
    kafka_broker = "localhost:9092"
    producer = KafkaProducer(bootstrap_servers=kafka_broker)

    with open("TurbineData/" + file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            message_data = ",".join(row)
            producer.send(topic, value=message_data.encode("utf-8"))
            time.sleep(1)  # Simulate real-time streaming

    producer.close()

if __name__ == "__main__":
    kafka_topics = ["turbine1", "turbine2", "turbine3", "turbine4", "turbine5"]
    file_paths = ["turbine1.csv", "turbine2.csv", "turbine3.csv", "turbine4.csv", "turbine5.csv"]

    with ThreadPoolExecutor(max_workers=len(kafka_topics)) as executor:
        executor.map(produce_to_kafka, kafka_topics, file_paths)