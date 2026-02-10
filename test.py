import pandas as pd
import numpy as np
antenna_filename = "sid_ant_text_latest.csv"
data = pd.read_csv(antenna_filename,delimiter = ';',header = 'infer')
print(data.info())
data['posx'] = np.array(data['posx'],dtype = float)*(-1)
data['posy'] =np.array(data['posy'],dtype = float)*(-1)
antenna_filename = "sid_ant_text_latest_test.csv"
data.to_csv(antenna_filename,sep= ';')
