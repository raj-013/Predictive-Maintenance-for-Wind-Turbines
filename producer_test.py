from kafka import KafkaProducer
import csv
import time
from concurrent.futures import ThreadPoolExecutor

def produce_to_kafka(topic):
    kafka_broker = "localhost:9092"
    producer = KafkaProducer(bootstrap_servers=kafka_broker)

    with open("test.csv", "r") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header
        header_sent = False
        for row in reader:
            if not header_sent:
                # Send headers only once
                message_data = ",".join(headers)
                producer.send(topic, value=message_data.encode("utf-8"))
                header_sent = True

            # Send row data
            message_data = ",".join(row)
            producer.send(topic, value=message_data.encode("utf-8"))
            time.sleep(1)

    producer.close()

if __name__ == "__main__":
    kafka_topics = ["topicTest"]

    with ThreadPoolExecutor(max_workers=len(kafka_topics)) as executor:
        executor.map(produce_to_kafka, kafka_topics)