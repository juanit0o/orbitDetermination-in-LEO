import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

full_dataUnpickled = pd.read_pickle("dataMultipleOrbits.pkl")
full_dataUnpickled = full_dataUnpickled.drop(columns=["xdotdot", "ydotdot", "zdotdot", "ecc",  "inc", "alt"], axis =1)
# full_dataUnpickled['target'] = full_dataUnpickled[['x', 'y', 'z', 'xdot', 'ydot', 'zdot']].values.tolist()
# full_dataUnpickled = full_dataUnpickled.drop(columns=["x", "y", "z", "xdot",  "ydot", "zdot"], axis =1)




#add datetime column from time column starting at 1 Jan 2020, starting at 0 for each orbit id
full_dataUnpickled['timeAux'] = full_dataUnpickled.groupby('orbit_id')['time'].transform(lambda x: x - x.min())
full_dataUnpickled['timeAux'] = full_dataUnpickled['timeAux'] + 1577836800
full_dataUnpickled['datetime'] = pd.to_datetime(full_dataUnpickled['timeAux'], unit='s')
#print orbit ids

##########################################
##########################################
#Change target to be a list of all x's, all y's, all z's, all xdot's, all ydot's, all zdot's

#new dataframe with number of lines = unique orbit ids, columns = starting time, list of all x for that orbit id, list of all y for that orbit id, list of all z for that orbit id, list of all xdot for that orbit id, list of all ydot for that orbit id, list of all zdot for that orbit id
full_dataUnpickled = full_dataUnpickled.groupby('orbit_id').agg({'time': 'first', 'x': list, 'y': list, 'z': list, 'xdot': list, 'ydot': list, 'zdot': list})
full_dataUnpickled = full_dataUnpickled.reset_index()
#join the lists of x, y, z, xdot, ydot, zdot into a list of lists
full_dataUnpickled['target'] = full_dataUnpickled[['x', 'y', 'z', 'xdot', 'ydot', 'zdot']].values.tolist()
#drop the columns x, y, z, xdot, ydot, zdot
full_dataUnpickled = full_dataUnpickled.drop(columns=["x", "y", "z", "xdot",  "ydot", "zdot"], axis =1)

#change the time column to datetime
# full_dataUnpickled['time'] = pd.to_datetime(full_dataUnpickled['time'], unit='s')
#make the time start in 1 Jan 2020
from datetime import datetime
full_dataUnpickled['time'] = datetime(2020, 1, 1)

#print size of the first target
# print(np.shape(full_dataUnpickled['target'][0]))
# print(np.shape(full_dataUnpickled['target'][1]))
print(full_dataUnpickled.head())
print(np.shape(full_dataUnpickled["target"][0][0]))
# print(full_dataUnpickled.head())
# #print shape of target of first orbit id
# print(np.shape(full_dataUnpickled['x'][0]))
# plt.plot(full_dataUnpickled['x'][0])
# plt.show()
#print datetime for orbit id = 0
# print(full_dataUnpickled[full_dataUnpickled['orbit_id'] == "1"]['datetime'])
