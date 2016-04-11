import CNN_MOD_4
import input_data_window_large
import CNN_STATIC_VARIABLES

subjects = ["01A","02A","03A","04A","05A","06A","08A","09A","10A","11A","12A","13A","14A","15A","16A","18A","19A","20A","21A","22A","23A"]



for subject_directory in subjects:
		#Generate one split array for each subject

		print "Subject: " + subject_directory

		
		print "Removing activities"
		activities = [0,3,9,12,15,16,17]

		filepath = '../../../../Prosjektoppgave/Notebook/data/'+subject+'/RAW_SIGNAL/'
  		files =   ['_Axivity_BACK_Back.csv', '_Axivity_THIGH_Right.csv','_GoPro_LAB_All.csv']
  		df_chest = pd.read_csv(filepath, header=None, sep='\,')
		df_thigh = pd.read_csv(filepath, header=None, sep='\,')
		df_label = pd.read_csv(filepath, header=None, sep='\,')
		#REMOVE {3:3, 9:9, 12:12, 15:15, 16:16, 17:17
		#Change 11-->10 (picking to bending)

		print "Create sliding windows"

		#Find mean for all sliding windows


