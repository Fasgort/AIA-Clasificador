import csv

from datetime import datetime
import numpy as np
import json


def load_no_show_issue(path='./data/No-show-Issue-Comma-300k.csv'):
    week_days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')

    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        feature_names = next(spamreader)
        feature_names.pop(5)
        target_names = ('No-Show', 'Show-Up')
        feature_names.pop(3)
        feature_names.pop(2)
        feature_names.insert(2, 'DayOfTheYe')
        data = list()
        target = list()
        for row in spamreader:
            instance = list()
            try:
                instance.append(validate_unsigned_integer(row, 0))  # Age
                instance.append(1 if row[1] == 'F' else 0)  # Gender
                # instance.append(row[2])    # AppointmentRegistration
                # instance.append(row[3])    # ApointmentData
                appoint_reg_dt = validate_string_datetime(row, 2)
                appoint_dt = validate_string_datetime(row, 3)
                app_doy = appoint_dt.timetuple().tm_yday
                instance.append(app_doy)  # Appointment day of the year
                instance.append(week_days.index(row[4]))  # DayOfTheWeek
                target.append(1 if row[5][0] == 'S' else 0)  # Status
                instance.append(int(row[6]))  # Diabetes
                instance.append(int(row[7]))  # Alcoolism
                instance.append(validate_boolean(row, 8))  # HiperTension
                instance.append(int(row[9]))  # Handcap
                instance.append(int(row[10]))  # Smokes
                instance.append(validate_boolean(row, 11))  # Scholarship
                instance.append(int(row[12]))  # Tuberculosis
                instance.append(int(row[13]))  # Sms_Reminder
                instance.append(abs(int(row[14])))  # AwaitingTime
                data.append(tuple(instance))
            except ValueError as ex:
                print(ex)
    with open(path[:-4] + 'cache.json', 'w') as outfile:
        json.dump((data, target, feature_names, target_names), outfile)
    return np.array(data, dtype='float_'), np.array(target, dtype='int_'), np.array(feature_names,
                                                                                    dtype='U10'), np.array(target_names,
                                                                                                           dtype='U10')


def diff_dt(dt_1, dt_2):
    if dt_2 >= dt_1:
        return (dt_2 - dt_1).days
    else:
        raise ValueError("Invalid datetime period: {} - {}".format(dt_2, dt_1))


def validate_boolean(row_data, index):
    if 0 <= int(row_data[index]) <= 1:
        return row_data[index]
    else:
        raise ValueError("Invalid boolean at {}: {}".format(index, row_data))


def validate_unsigned_integer(row_data, index):
    if 0 <= int(row_data[index]):
        return int(row_data[index])
    else:
        raise ValueError("Invalid u integer at {}: {}".format(index, row_data))


def validate_string_datetime(row_data, index):
    try:
        dt = datetime.strptime(row_data[index], '%Y-%m-%dT%H:%M:%SZ')
        return dt
    except ValueError:
        raise ValueError("Invalid datetime string: {}".format(dt))


def load_no_show_issue_cache(path='./data/No-show-Issue-Comma-300kcache.json'):
    with open(path, 'r') as jsonfd:
        raw = json.load(jsonfd)
        return np.array(raw[0], dtype='float_'), np.array(raw[1], dtype='int_'), np.array(raw[2],
                                                                                          dtype='U10'), np.array(raw[3],
                                                                                                                 dtype='U10')


def filter_features(x_data, x_names, features_2_allow):
    res_data = np.copy(x_data)
    res_names = [feature for feature in x_names if feature in features_2_allow]
    i_2_del = [i for i, feature in enumerate(x_names) if feature not in features_2_allow]
    return np.delete(res_data, i_2_del, axis=1), res_names
