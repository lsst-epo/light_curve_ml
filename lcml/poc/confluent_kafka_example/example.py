#!/usr/bin/env python3
from confluent_kafka import avro, Consumer, KafkaError, Producer
from confluent_kafka.avro import AvroConsumer, AvroProducer
from confluent_kafka.avro.serializer import SerializerError


def testProducer(topic):
    p = Producer({'bootstrap.servers': "localhost:9092"})
    stuff = ['a', 'b', 'c'] * 10
    for data in stuff:
        p.produce(topic, data.encode('utf-8'))
    p.flush()


def testConsumer(topic):
    kwargs = {'bootstrap.servers': "localhost:9092", 'group.id': 'mygroup',
              'default.topic.config': {'auto.offset.reset': 'smallest'}}
    c = Consumer(kwargs)
    c.subscribe([topic])
    running = True
    while running:
        msg = c.poll()
        if not msg.error():
            print('Received message: %s' % msg.value().decode('utf-8'))
        elif msg.error().code() != KafkaError._PARTITION_EOF:
            print(msg.error())
            running = False
    c.close()


def testAvroProducer(topic):
    value_schema = avro.load('ValueSchema.avsc')
    key_schema = avro.load('KeySchema.avsc')
    value = {"name": "Value"}
    key = {"name": "Key"}
    config = {'bootstrap.servers': "localhost:9092",
              'schema.registry.url': 'http://schem_registry_host:port'}
    avroProducer = AvroProducer(config,
                                default_key_schema=key_schema,
                                default_value_schema=value_schema)
    avroProducer.produce(topic=topic, value=value, key=key)
    avroProducer.flush()


def testAvroConsumer(topic):
    config = {'bootstrap.servers': "localhost:9092", 'group.id': 'groupid',
              'schema.registry.url': 'http://127.0.0.1:8081'}
    c = AvroConsumer(config)
    c.subscribe([topic])
    running = True
    msg = None
    while running:
        try:
            msg = c.poll(10)
            if msg:
                if not msg.error():
                    print(msg.value())
                elif msg.error().code() != KafkaError._PARTITION_EOF:
                    print(msg.error())
                    running = False
        except SerializerError as e:
            print("Message deserialization failed for %s: %s" % (msg, e))
            running = False

    c.close()


def main():
    topic = "testTopic"
    testProducer(topic)
    testConsumer(topic)


if __name__ == "__main__":
    main()
