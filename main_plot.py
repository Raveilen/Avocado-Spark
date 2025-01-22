from kafka_handler import KafkaRecive
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import json

kafka_consumer = KafkaRecive()

print("Start listening...")

counter = 0
x_date = []
y_real_data = []
y_predict_data = []

# Set plot size
plt.figure(figsize=(12,8))

for val in kafka_consumer.receive_message():

    #print(f' yield val : {val}')

    data = json.loads(val)

    if counter > 15:
        break

    counter += 1

    x_date.append(data['Date'])
    y_real_data.append(data['avgAveragePrice'])
    y_predict_data.append(data['avg_prediction'])

    #print(f" data : {data['Date']}  ;  rel : {data['avgAveragePrice']} ;  predict : {data['avg_prediction']}")
    
    # Draw plot
    plt.plot(x_date, y_real_data, marker = 'o', color = 'red')
    plt.plot(x_date, y_predict_data, marker = '^', color = 'blue')

    # Set plot legend
    plt.legend(["Average price", "Predicted average"], loc="upper right")
    
    # Set plot attributes
    plt.title("Avocado prices over time")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.ylabel("Price of avocado")

    # Draw plot
    plt.draw()
    plt.pause(0.4)
    plt.clf()


kafka_consumer.close()
