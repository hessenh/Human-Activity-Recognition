import CNN_MOD_4
import input_data_window_large
import CNN_STATIC_VARIABLES
import pandas as pd
import numpy as np

VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
subject_set = VARS.get_subject_set(False)

filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/GoPro_LAB_All_L.csv'
labels = pd.read_csv(filepath, header=None, sep='\,',engine='python')

filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/Axivity_BACK_Back_X.csv'
X_BACK = pd.read_csv(filepath, header=None, sep='\,',engine='python')
filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/Axivity_BACK_Back_Y.csv'
Y_BACK = pd.read_csv(filepath, header=None, sep='\,',engine='python')
filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/Axivity_BACK_Back_Z.csv'
Z_BACK = pd.read_csv(filepath, header=None, sep='\,',engine='python')

filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/Axivity_THIGH_Right_X.csv'
X_THIGH = pd.read_csv(filepath, header=None, sep='\,',engine='python')
filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/Axivity_THIGH_Right_Y.csv'
Y_THIGH = pd.read_csv(filepath, header=None, sep='\,',engine='python')
filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/DATA_WINDOW/3.0/ORIGINAL/Axivity_THIGH_Right_Z.csv'
Z_THIGH = pd.read_csv(filepath, header=None, sep='\,',engine='python')


filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/RAW_SIGNALS/'+subject_set[0][2]+'_Axivity_THIGH_Right.csv'
ORIGINAL_THIGH = pd.read_csv(filepath, header=None, sep='\,',engine='python')
filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/RAW_SIGNALS/'+subject_set[0][2]+'_Axivity_BACK_Back.csv'
ORIGINAL_BACK = pd.read_csv(filepath, header=None, sep='\,',engine='python')
filepath = '../../../../../Prosjektoppgave/Notebook/data/'+subject_set[0][2]+'/RAW_SIGNALS/'+subject_set[0][2]+'_GoPro_LAB_All.csv'
ORIGINAL_LABEL = pd.read_csv(filepath, header=None, sep='\,',engine='python')


print len(labels)
print len(ORIGINAL_LABEL)

real_windows = np.array([])
energy_windows = np.array([])
diff_windows = np.array([])

for i in range(0,len(ORIGINAL_LABEL)-1):
	presLabel = ORIGINAL_LABEL.iloc[i][0]
	nextLabel = ORIGINAL_LABEL.iloc[i+1][0]
	if nextLabel != presLabel:
		real_windows = np.append(real_windows,i+2)



def extract_energy_feature(df_x, df_y, df_z, start, length, feature_type, sensor):    
    data_frame_result = ((np.sqrt(np.square(df_x.sub(np.mean(df_x,axis=1),axis=0)).sum(axis=1)) + np.sqrt(np.square(df_y.sub(np.mean(df_y,axis=1),axis=0)).sum(axis=1)) + np.sqrt(np.square(df_z.sub(np.mean(df_z,axis=1),axis=0)).sum(axis=1)))/3.0)/len(df_x.iloc[0].values)
    data_frame_result = pd.DataFrame(np.array(data_frame_result))
    data_frame_result.columns = [feature_type + '_' + sensor]
    return data_frame_result


energy_THIGH = extract_energy_feature(X_THIGH,Y_THIGH,Z_THIGH,0,100,'energy','thigh')
energy_BACK = extract_energy_feature(X_BACK,Y_BACK,Z_BACK,0,100,'energy','back')

i=0
while i < len(energy_THIGH)-3:
	pres_energy_thigh = energy_THIGH.iloc[i][0]
	next_energy_thigh = energy_THIGH.iloc[i+3][0]
	pres_energy_back = energy_BACK.iloc[i][0]
	next_energy_back = energy_BACK.iloc[i+3][0]
	if np.absolute(np.subtract(pres_energy_thigh,next_energy_thigh)) > 0.01 and np.absolute(np.subtract(pres_energy_back,next_energy_back)) > 0.01:
		energy_windows = np.append(energy_windows,i*100 + 300)
		i=i+3
	else:	
		i=i+1
		 




k=0
for i in range(0,len(energy_windows)):
	found=False
	diff = np.absolute(np.subtract(real_windows[k],energy_windows[i]))
	while found == False and k<len(real_windows)-1:
		newDiff = np.absolute(np.subtract(real_windows[k],energy_windows[i]))
		if newDiff<=diff:
			diff=newDiff
			k=k+1
		else:
			k=k-1
			print 'diff: ', diff
			print 'energy window:', energy_windows[i]
			print 'real window: ',real_windows[k]
			found = True
			diff_windows = np.append(diff_windows,diff)	

#print energy
print 'length energy windows: ',len(energy_windows)
print 'length real windows: ' , len(real_windows)
print 'andel vindu: ', len(energy_windows)/len(real_windows)
print 'sum diff: ', np.sum(diff_windows)/len(energy_windows)
print 'sum median: ', np.median(diff_windows)
print 'average real window size: ', len(ORIGINAL_LABEL)/len(real_windows)

np.savetxt('./real_windows.csv', real_windows, delimiter=",")
np.savetxt('./energy_windows.csv', energy_windows, delimiter=",")