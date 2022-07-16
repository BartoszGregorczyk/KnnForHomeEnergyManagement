import json
from this import s
from turtle import st, up
from xmlrpc.client import boolean
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (KNeighborsClassifier)
import pickle

def get_data_from_json(file_name):
    data = open(file_name, 'r')
    return json.load(data)

def save_data_to_json(file_name, data):
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

def calculate_time_window_photovoltaics_generation(weather_data, generation_power):
    sum_of_energy_in_kwh = 0.0
    for value in weather_data:
        solar_radiation = value['forecast']
        diff_in_hours = 0.0833
        solar_radiation_coefficient = solar_radiation / 1000
        output_power = generation_power * solar_radiation_coefficient * (1 - 0.05)
        output_power_in_kwh = output_power / 1000 * diff_in_hours
        sum_of_energy_in_kwh += output_power_in_kwh
    return sum_of_energy_in_kwh

def create_report_with_class(single_raport, obj_class):
    
    #------------------------------getting fields from reports------------------------------
    energy_storage_after_first_window = list(single_raport['energy_storage'].values())[0]
    total_storage_capacity = single_raport['total_storage_capacity']
    energy_generation_first_window = list(single_raport['energy_generation'].values())[0]
    energy_usage_first_window = list(single_raport['energy_usage'].values())[0]
    surplus_after_first_window = list(single_raport['surplus_data'].values())[0]
    used_public_grid_in_first_window = list(single_raport['public_grid_data'].values())[0]
    exchange_price_at_the_beginning = list(single_raport['exchange_data'].values())[0]
    initial_surplus_value = single_raport['initial_grid_surplus_value']
    initial_storage_value = single_raport['initial_storage_charge_value']
    generation_power = single_raport['generation_power']
    weather_data_third_window = list(single_raport['weather_data'].values())[-13:]
    third_window_generation = calculate_time_window_photovoltaics_generation(weather_data_third_window, generation_power)
    #-------------------------------------------------------------------------------------

    battery_charge = energy_storage_after_first_window / (total_storage_capacity + 0.00001), #this small value is to protect against division by zero
    battery_charge = battery_charge[0]

    generation_to_usage_ratio = energy_generation_first_window / (energy_usage_first_window + 0.00001)
    initial_suprlus_and_storage_to_usage_ratio = (initial_surplus_value + initial_storage_value) / (energy_usage_first_window + 0.00001)
    
    if_taken_from_public_grid = 0
    if (used_public_grid_in_first_window != 0):
        if_taken_from_public_grid = 1

    if_taken_from_storage = 0
    if (initial_storage_value > energy_storage_after_first_window):
        if_taken_from_storage = 1

    if_taken_from_surplus = 0
    if (initial_surplus_value > surplus_after_first_window):
        if_taken_from_surplus = 1

    report2 = {
                'battery_charge' :  battery_charge,
                'generation_to_usage_ratio' : generation_to_usage_ratio,
                'initial_suprlus_and_storage_to_usage_ratio' : initial_suprlus_and_storage_to_usage_ratio,
                'if_taken_from_public_grid' : if_taken_from_public_grid,
                'exchange_price' : exchange_price_at_the_beginning,
                'if_taken_from_storage' : if_taken_from_storage,
                'if_taken_from_surplus' : if_taken_from_surplus,
                'third_window_generation' :third_window_generation,
                'class': obj_class}

    report = {
                'energy_generation' : list(single_raport['energy_generation'].values())[0],
                'energy_usage' : list(single_raport['energy_usage'].values())[0],
                'energy_storage' : list(single_raport['energy_storage'].values())[0],
                'public_grid_data' : list(single_raport['public_grid_data'].values())[0],
                'surplus_data' : list(single_raport['surplus_data'].values())[0],
                'exchange_data' : list(single_raport['exchange_data'].values())[0],
                'class': obj_class}
    return report2

def get_reports_with_appropriate_classes(all_raports, classes, how_many_reports):
    raports_with_classes = []

    main_loop_iterations = how_many_reports
    if (how_many_reports == 0):
        main_loop_iterations = len(all_raports)

    for number_of_raport in range(0, main_loop_iterations):
        single_raport = all_raports[number_of_raport]
        grid_price = single_raport["public_grid_price"]
        exchange_price = list(single_raport["exchange_data"].values())[11] # Exchange energy price at the end of first window
        third_window_energy_from_public_grid = list(single_raport["public_grid_data"].values())[2]

        if exchange_price < grid_price:
            if third_window_energy_from_public_grid < 0.001:
                raport_with_class = create_report_with_class(single_raport, '0.0')
                raports_with_classes.append(raport_with_class)
            else:
                for index in range (0, len(classes) - 1):
                    down_range = float(classes[index])
                    up_range = float(classes[index + 1])
                    if down_range < third_window_energy_from_public_grid <= up_range:
                        down_diff = abs(down_range - third_window_energy_from_public_grid)
                        up_diff = abs(up_range - third_window_energy_from_public_grid)
                        if down_diff < up_diff:
                            raport_with_class = create_report_with_class(single_raport, classes[index])
                            raports_with_classes.append(raport_with_class)
                            break
                        else:
                            raport_with_class = create_report_with_class(single_raport, classes[index + 1])
                            raports_with_classes.append(raport_with_class)
                            break
                    elif (index == len(classes) - 2):
                        raport_with_class = create_report_with_class(single_raport, classes[index + 1])
                        raports_with_classes.append(raport_with_class) 
        else:
            raport_with_class = create_report_with_class(single_raport, '0.0')
            raports_with_classes.append(raport_with_class)
    return raports_with_classes

def show_error_rate(train_x, train_y, test_x, test_y):
    error = []

    for i in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x, train_y)
        pred_i = knn.predict(test_x)
        error.append(np.mean(pred_i != test_y))

    plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

def create_classes_values(initial_value, max_value, step):
    classes = []
    for i in np.arange(initial_value, max_value + step, step):
        classes.append(str(round(i,2)))
    return classes


def main():
    NUMBER_OF_DATA_COLUMS_DATASET = 8

    #---------------------------------set up parameters----------------------------------------------------------------
    nr_of_reports = int(input('enter the number of reports for the algorithm (type 0 for max): '))
    initial_value_of_class = float(input('enter the first float value (e.g. 0.0 - one digit after decimal point) for KNN class: '))
    max_value_of_class = float(input('enter the last float value (e.g. 0.0 - one digit after decimal point) for KNN class: '))
    step_of_class = float(input('enter the step value (e.g. 0.0 - one digit after decimal point) for KNN class: '))
    if_show_neigbours_error_rate = input('type "1" when you want to show neighbours error rate plot or enter if not: ')
    classes = create_classes_values(initial_value_of_class, max_value_of_class, step_of_class)
    #-------------------------------------------------------------------------------------------------------------------
   
    all_data = get_data_from_json('random_data_6.json')
    data = get_reports_with_appropriate_classes(all_data, classes, nr_of_reports)
    
    dataset =  pd.DataFrame.from_dict(data, orient='columns')
    dataset.to_csv(r'datasets\data.csv')
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, NUMBER_OF_DATA_COLUMS_DATASET].values   

    # le = LabelEncoder()
    # X[:,0] = le.fit_transform(X[:,0]) #TODO THINK ABOUT IT!!!!  probably it is not necessary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4) #random = None - for every run -> random data
    sc = StandardScaler() 


    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test) 

    
    classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2) # TODO Think about this arguments in method KNN, read about it
    classifier.fit(X_train, y_train)

    # save the model to disk
    # filename = 'knn_model.sav'
    # pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.predict(X_test)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test,y_pred)

    print('how much we should buy', y_test)
    print ('predicted how much we should buy: ', y_pred)
    # print ('predicted how much we should buy (from file): ', result)
    print('accuracy ', ac)
    
    diff = 0
    diff_temp = 0
    for i in range(0, len(y_test)):
        diff_temp = abs(float(y_test[i]) - float(y_pred[i]))
        diff += diff_temp
    print('mean error: ', diff / len(y_test))

    temp = 0
    for i in range(0, len(classes)):
        for j in range(0, len(data)):
            if data[j]['class'] == classes[i]:
                temp += 1
        print('class ', classes[i], ' number of raports: ', temp)
        temp = 0

    if (if_show_neigbours_error_rate == '1'):
        show_error_rate(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main()
