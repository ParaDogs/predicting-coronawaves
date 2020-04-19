import neuro
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
import os
import data_collector
import time
import datetime

earth_population = 8000
zero_human = 1/(earth_population*2)
doomsday = 1735689600
max_gini_index = 70
min_lat = -90
max_lat = 90
min_lng = -180
max_lng = 180

covid_reports_files = './data/datasets/'
files = os.listdir(covid_reports_files)

input_nodes = 9 # время, заболевших, умерших, выздоровевших, уровень жизни, широта, долгота, население, соседи? 
hidden_nodes = 200 # экспериментально
output_nodes = 3 # множители заболевших, умерших, выздоровевших
learning_rate = 0.15 # экспериментально

if os.path.isfile('neuro.pickle'):
	with open('neuro.pickle', 'rb') as f:
		n = pickle.load(f)
else:
	n = neuro.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
	collector = data_collector.Country_info_collector()
	epochs = 2
	for e in range(epochs):
		for file_idx in range(len(files)-1):
			for country_idx in range(len(collector.countries)):
				country_name = collector.countries[country_idx]['name']
				next_time_str = files[file_idx+1].split('.')[0]
				now_time_str = files[file_idx].split('.')[0]
				now_time = time.mktime(datetime.datetime.strptime(now_time_str, "%m-%d-%Y").timetuple())/doomsday
				gini = collector.gini(country_name)
				print(gini)
				gini_ratio = gini/max_gini_index
				lat = (collector.latlng(country_name)[0] - min_lat)/(max_lat - min_lat)
				lng = (collector.latlng(country_name)[1] - min_lng)/(max_lng - min_lng)
				population = (collector.population(country_name))/earth_population
				borders_index = collector.borders(country_name, now_time_str)

				next_infected = collector.infected(country_name,next_time_str) + zero_human
				now_infected = collector.infected(country_name,now_time_str) + zero_human
				next_dead = collector.dead(country_name,next_time_str) + zero_human
				now_dead = collector.dead(country_name,now_time_str) + zero_human
				next_recovered = collector.recovered(country_name,next_time_str) + zero_human
				now_recovered = collector.recovered(country_name,now_time_str) + zero_human
				infected_ratio = (next_infected/now_infected)/earth_population + zero_human
				dead_ratio = (next_dead/now_dead)/earth_population + zero_human
				recovered_ratio = (next_recovered/now_recovered)/earth_population + zero_human
				
				inputs_data = [now_time, now_infected/earth_population, now_dead/earth_population, now_recovered/earth_population, gini_ratio, lat, lng, population ,borders_index]
				targets_data = [infected_ratio, dead_ratio, recovered_ratio]

				inputs = np.asfarray(inputs_data)
				targets = np.asfarray(targets_data)

				n.train(inputs, targets)
		with open('neuro.pickle', 'wb') as f:
			pickle.dump(n, f)
	
inputs_data = [1587040866/doomsday, 0.021032, 0.001020, 0.012, 0.4, 0.6, 0.8, 0.0004, 0.00003323]
inputs = np.asfarray(inputs_data)
outputs = n.query(inputs)
label = np.argmax(outputs)
print("ответ", label)