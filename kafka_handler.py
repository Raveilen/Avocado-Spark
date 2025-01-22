from kafka import KafkaProducer, KafkaConsumer
import json

class KafkaSend:
    
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers = ["localhost:29092"]
        )

    def send_message(self, message_value):
        self.producer.send(
            "avocado_price",
            json.dumps(message_value).encode('utf-8'),
        )

        print("KAFKA -> MSG send...")

    def close(self):
        self.producer.close()


class KafkaRecive:

    def __init__(self):
        self.consumer = KafkaConsumer(
            bootstrap_servers = ['localhost:29092'],
            api_version = "7.3.1",
            auto_offset_reset = 'latest',
            enable_auto_commit = False,
            value_deserializer = lambda x : json.loads(x.decode('utf-8'))
        )

        self.consumer.subscribe(topics = ['avocado_price'])


    def receive_message(self):
        for message in self.consumer:
            #print(f"RECEIVE - > {message.value}")
            yield message.value

    
    def close(self):
        self.consumer.close()
