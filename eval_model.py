import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def load_data(csv_files):

	dfs = []
	for csv_file in csv_files:
		df_ = pd.read_csv(csv_file)
		df = df_[df_.label < 3].reset_index(drop=True)
		df = df.drop(['dir_name','img_file'], axis=1)
		dfs.append(df)

	df = pd.concat(dfs)
	df = df.sample(frac=1)

	N = len(df)
	P_len = len(df[df.label == 2])
	N_len = len(df[df.label == 1])
	print("Ratio of 2:",P_len/N)
	print("Ratio of 1:",N_len/N)

	likelihood_dic=['Likelihood.UNKNOWN','Likelihood.VERY_UNLIKELY','Likelihood.UNLIKELY',
	'Likelihood.POSSIBLE','Likelihood.LIKELY','Likelihood.VERY_LIKELY']
	vec = lambda x: likelihood_dic.index(x)
	df.anger = df.anger.apply(vec)
	df.joy = df.joy.apply(vec)
	df.sorrow = df.sorrow.apply(vec)
	df.surprise = df.surprise.apply(vec)

	# Remove emotion features
	df = df.drop(['anger','joy','sorrow','surprise'], axis=1)

	df['x'] = (df.bound_left + df.bound_right)/2
	df['y'] = (df.bound_bot + df.bound_top)/2
	df['box_x'] = df.bound_right - df.bound_left
	df['box_y'] = df.bound_bot- df.bound_top

	df = df.drop(['bound_left','bound_top','bound_right','bound_bot'], axis=1)

	df_x = df.drop('label', axis=1)
	df_y = df.label

	global features
	features = df_x.columns.values

	return np.array(df_x.values), np.array(df_y.values)

def svm_model(train_X, train_Y):
	clf = svm.SVC()
	clf.fit(train_X, train_Y)
	return clf

def random_forest(train_X, train_Y):
	clf = RandomForestClassifier(n_estimators=15)
	clf.fit(train_X, train_Y)
	for colname, importance in zip(features, clf.feature_importances_):
		print("\t"+f'{colname:15} {importance}')
	return clf

def eval_model(model, test_X, test_Y):
	predict_Y = model.predict(test_X)
	TP, TN, FP, FN, N = 0, 0, 0, 0, len(test_Y)
	for i in range(N):
		if predict_Y[i] == 2:
			if test_Y[i] == 2:
				TP += 1
			else:
				FP += 1
		else:
			if test_Y[i] == 2:
				FN += 1
			else:
				TN += 1
	print("-- [TP,TN,FP,FN] =", [TP,TN,FP,FN])
	print("-- Accuracy:\t", (TP+TN)/N)
	print("-- Precision:\t", (TP)/(TP+FP))
	print("-- Recall:\t", (TP)/(TP+FN))
	print("-- F1:\t\t", (2*TP)/(2*TP+FN+FP))
	print("")

# Cross Validation for all data
def run(csv_files, model):
	print("Classify with",model.__name__,"\n")

	X, Y = load_data(csv_files)
	N = len(X)
	fold = 5
	unit = int(N / fold) + 1
	test_index = 0
	cv_count = 1
	while test_index < N:
		test_size = min(unit, N - test_index)
		train_X = np.concatenate([X[0:test_index],X[test_index + test_size:N]], axis=0)
		train_Y = np.concatenate([Y[0:test_index],Y[test_index + test_size:N]], axis=0)
		test_X = X[test_index:test_index + test_size]
		test_Y = Y[test_index:test_index + test_size]

		print("## Cross Validation ("+str(cv_count)+"/"+str(fold)+")")
		eval_model(model(train_X, train_Y), test_X, test_Y)

		test_index += test_size
		cv_count += 1

# Train / Test for data from different classes
def run2(csv_files, model):
	print("Classify with",model.__name__,"\n")

	file_class = [1,1,2,3]
	for i in range(min(file_class), max(file_class)+1):
		train_files, test_files = [], []
		for j in range(len(file_class)):
			if file_class[j] == i:
				test_files.append(csv_files[j])
			else:
				train_files.append(csv_files[j])

		print("## Test for class",i)

		train_X, train_Y = load_data(train_files)
		test_X, test_Y = load_data(test_files)

		eval_model(model(train_X, train_Y), test_X, test_Y)

def main():
	path = '../../Google Drive/Documents/Lectures/0_IntelligentUI/02 Project/05_Dataset/labeled/'
	csv_files = []
	csv_files.append(path + "detection_0900_yj_labeled.csv")
	csv_files.append(path + "detection_0900_tg_labeled.csv")
	csv_files.append(path + "detection_1030_5sec_labeled.csv")
	csv_files.append(path + "detection_1430_labeled.csv")
	run(csv_files, random_forest)
	# run2(csv_files, random_forest)

if __name__ == '__main__':
    main()