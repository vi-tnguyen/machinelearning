import matplotlib.pylab as plt
import pandas as pd
import json

def read(filename, file_format, json_obj_top_level_id = None):
	'''
	inputs:
		filename = name of ifle
		file_format = type of format, currently can be 'csv' or 'json objects' 
	outputs:
		dataset in pandas dataframe format
	'''
	df = []
	if file_format == 'csv':
		df = pd.read_csv(filename)
	elif file_format == 'json objects':

		with open(filename) as data_file:
			data_dict = {}

			for line in data_file:
				row = json.loads(line)
				top_id_val = row.pop(json_obj_top_level_id, None)
				data_dict[top_id_val] = row
		df = pd.DataFrame.from_dict(data_dict, orient = "index")
		# resets df index to integers, instead of the top_id_val
		df = df.reset_index().rename(columns={"index": json_obj_top_level_id})
	else:
		print('the current file format is not supported')

	return df