import io
import csv
import os

from google.cloud import vision
from google.oauth2 import service_account

vision_client = vision.Client(
    project='Oing',
    credentials=service_account.Credentials.from_service_account_file(
        'Oing-b491692df533.json'
    )
)

FACE_DETECTION_MAX_FACES = 32

def detect_faces(root_path, class_path, file):
	if not os.path.isfile(root_path+class_path+file):
		return None
	with io.open(root_path+class_path+file, 'rb') as image_file:
		content = image_file.read()
	image = vision_client.image(content=content)

	faces = image.detect_faces(limit=FACE_DETECTION_MAX_FACES)

	detections = []
	for face in faces:
		bound = [[bound.x_coordinate,bound.y_coordinate] for bound in face.bounds.vertices]
		detections.append([class_path, file]+bound[0] + bound[2]
			+ [face.angles.pan, face.angles.roll, face.angles.tilt]
			+ [face.anger, face.joy, face.sorrow, face.surprise])

	return detections

def main():
	dir_root = 'data_gopro/filtered/'
	dir_class = ['170524_0900_101/','170524_1030_101/','170524_1430_101/']
	start_file = [11107, 16102, 19198]
	end_file = [15363, 18788, 20780]
	with open('detection.csv', 'w') as csv_file:
		fieldnames=['dir_name','img_file','bound_left','bound_top','bound_right','bound_bot','pan','roll','tilt','anger','joy','sorrow','surprise']
		csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
		csv_writer.writerow(fieldnames)
		for i in range(2,3):
			current_file = start_file[i]
			while current_file <= end_file[i]:
				img_file = 'G00'+str(current_file)+'.jpg'
				current_file += 1
				face_detections = detect_faces(dir_root, dir_class[i], img_file)
				if face_detections != None:
					csv_writer.writerows(face_detections)
					print('Class ' + str(i+1) + ': ' + str(current_file - start_file[i]) + '/' + str(end_file[i] - start_file[i] + 1))
				else:
					print('Class ' + str(i+1) + ': ' + str(current_file - start_file[i]) + '/' + str(end_file[i] - start_file[i] + 1) + ' (no file)')

def main2():
	with open('detection.csv','wb') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',')
		csv_writer.writerow(['filename','bound_left'])

if __name__ == '__main__':
    main()
