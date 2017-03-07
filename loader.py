import csv

import numpy as np


def load_no_show_issue():
    week_days= ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
    with open('./data/No-show-Issue-Comma-300k.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        feature_names = next(spamreader)
        target_name = feature_names.pop(5)
        feature_names.pop(3)
        feature_names.pop(2)
        data= list()
        target= list()
        for row in spamreader:
            instance = list()
            instance.append(int(row[0]))    # Age
            instance.append(1 if row[1] == 'F' else 0) #Gender
            #instance.append(row[2])    # AppointmentRegistration
            #instance.append(row[3])    # ApointmentData
            instance.append(week_days.index(row[4]))    # DayOfTheWeek
            target.append(1 if row[5][0] == 'S' else 0) # Status
            instance.append(int(row[6]))    # Diabetes
            instance.append(int(row[7]))    # Alcoolism
            instance.append(int(row[8]))    # HiperTension
            instance.append(int(row[9]))    # Handcap
            instance.append(int(row[10]))   # Somkes
            instance.append(int(row[11]))   # Scholarship
            instance.append(int(row[12]))   # Tuberculosis
            instance.append(int(row[13]))    # Sms_Reminder
            #instance.append(int(row[14]))  # AwaitingTime
            data.append(tuple(instance))

    return np.array(data), target, np.array(feature_names),  [target_name]
