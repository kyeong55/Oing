import matplotlib.pyplot as plt
import csv
import numpy as np

def moving_average(values, window = 10):
	weight = np.repeat(1.0, window)/window
	return np.convolve(values, weight, 'valid')

def extract(csv_file):
	face_num = []
	attention_num = []
	limit = 50000
	with open(csv_file) as f:
		reader = csv.reader(f, delimiter=',', lineterminator='\n')
		next(reader)
		img_file = ""
		img_index = -1
		for row in reader:
			if row[1] != img_file:
				limit -= 1
				if limit < 0:
					break
				img_file = row[1]
				face_num.append(0)
				attention_num.append(0)
				img_index += 1
			face_num[img_index] += 1
			if row[13] == '2':
				attention_num[img_index] += 1
	return np.array(face_num), np.array(attention_num)

def main():
	csv_path = '../../Google Drive/Documents/Lectures/0_IntelligentUI/02 Project/05_Dataset/labeled/'
	img_path = '../../Google Drive/Documents/Lectures/0_IntelligentUI/02 Project/05_Dataset/image_files/'
	csv_file = csv_path + "detection_1030_5sec_labeled.csv"
	csv_file = csv_path + "detection_1430_labeled.csv"
	smoothing_window = 10
	face_num, attention_num = extract(csv_file)
	diff_num = face_num - attention_num
	x_axis = np.array(range(len(face_num)-smoothing_window+1)) * 5
	plot1, = plt.plot(x_axis, moving_average(face_num,smoothing_window), label='Detected faces')
	plot2, = plt.plot(x_axis, moving_average(attention_num,smoothing_window), label='Watching faces')
	plt.xlabel('time(sec)')
	plt.ylabel('# of people')
	plt.legend(handles = [plot1,plot2])
	plt.show()

if __name__ == '__main__':
    main()