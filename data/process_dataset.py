import os
import re
from shutil import copyfile
import csv
import argparse


row = ['image_name', 'label']

# Replace the name of directories with your name of directory
directories = ['bangla', 'devanagari', 'telugu']

def ensure_directory(path):
	""" 
	This function creates a directory if does not exists  
	Args:
		path (string): directory location
	"""
	if not os.path.exists(path):
		os.makedirs(path)
	return True
	
def sorted_nicely(l):
	""" Imported from stackoverflow for telugu dataset to sort alphanumeric filenames properly  
	Sorts the given iterable in the way that is expected.  
	Args:
		l (iterable): The iterable to be sorted.
	"""
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key = alphanum_key)

def process_dataset(directories=directories, to_csv=True):
	""" Read dataset & arrange into 0-9 label directories  
	Args:
		directories (list): List of data directories downloaded.
		csv (bool): Creates a csv file instead of sorting into folders
	Returns:
		bool: nothing important
	"""
	# Relative path things
	rel_dirname = os.path.dirname(__file__)

	if to_csv:
		# Create folder for storing csv files
		ensure_directory(os.path.join(rel_dirname, 'csv'))

		# Read files sequentially & move into CSV files for each directory & file
		for dirname in directories:
			with open(os.path.join(rel_dirname, 'csv/'+dirname+'.csv'), 'w') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
				print('Reading', dirname, 'folder...')
				digit = 0
				dir_path = os.path.join(rel_dirname, 'digits/'+dirname)
				for filename in sorted_nicely(os.listdir(dir_path)):
					if filename.endswith('.bmp'):
						# Write the filename and label
						writer.writerow([filename, digit])
						digit += 1
						if digit%10 == 0:
							digit = 0
				print('Finished reading', dirname, 'folder...')
	else:
		# Create training folders 
		ensure_directory(os.path.join(rel_dirname, 'training'))

		# Read files sequentially & move into train & test folders as per labels
		for dirname in directories:
			print('Reading', dirname, 'folder...')
			digit = 0
			dir_path = os.path.join(rel_dirname, 'digits/'+dirname)
			for filename in sorted_nicely(os.listdir(dir_path)):
				if filename.endswith('.bmp'):
					ensure_directory(os.path.join(rel_dirname, 'training/'+dirname+'/'+str(digit)))
					# Move the file
					copyfile(os.path.join(rel_dirname,'digits/'+dirname+'/'+filename), os.path.join(rel_dirname,'training/'+dirname+'/'+str(digit)+'/'+filename))
					digit += 1
					if digit%10 == 0:
						digit = 0
			print('Finished reading', dirname, 'folder...')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', dest='to_csv', action='store_true')
	args = parser.parse_args()
	process_dataset(directories=directories, to_csv=args.to_csv)
