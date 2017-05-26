import io

from google.cloud import vision
from google.oauth2 import service_account

import tkinter as tk
from PIL import ImageTk, Image, ImageDraw

vision_client = vision.Client(
    project='Oing',
    credentials=service_account.Credentials.from_service_account_file(
        'Oing-b491692df533.json'
    )
)

FACE_DETECTION_MAX_FACES = 32
GUI = None

def detect_faces(path):
    """Detects faces in an image."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision_client.image(content=content)

    faces = image.detect_faces(limit=FACE_DETECTION_MAX_FACES)
    print('Faces:')

    angles = []
    bounds = []

    for face in faces:
        # print('pan: {}'.format(face.angles.pan))
        # print('roll: {}'.format(face.angles.roll))
        # print('tilt: {}'.format(face.angles.tilt))
        angle = {}
        angle['pan'] = face.angles.pan
        angle['roll'] = face.angles.roll
        angle['tilt'] = face.angles.tilt
        bound = [(bound.x_coordinate,bound.y_coordinate) for bound in face.bounds.vertices]
        angles.append(angle)
        bounds.append(bound)

        print('face bounds: ',bound)

    return bounds, angles

class LabelingGUI:
	def __init__(self):
		self.ori_img_w = 1200
		self.ori_img_h = 900
		self.ori_frag_w = self.ori_img_w + 100
		self.ori_frag_h = self.ori_img_h + 100

		self.face_img_w = 200
		self.face_img_h = 300
		self.face_frag_w = self.face_img_w + 100
		self.face_frag_h = self.face_img_h + 100

		self.root = root = tk.Tk()
		self.root.title("Oing Labeling")
		self.root.bind('<KeyPress>', onKeyPress)

		self.frame1 = tk.Frame(self.root)
		self.frame1.pack(side=tk.LEFT)
		self.frame2 = tk.Frame(self.root)
		self.frame2.pack(side=tk.LEFT)

		self.cv1 = tk.Canvas(self.frame1,width=self.ori_frag_w,height=self.ori_frag_h)
		self.cv1.pack(side=tk.TOP)

		self.img_name = tk.Label(self.frame2, padx=80, pady=12, font=(None,20, 'bold'))
		self.img_name.pack(side=tk.TOP)
		self.face_index = tk.Label(self.frame2, pady=8, font=(None,16,'bold'))
		self.face_index.pack(side=tk.TOP)

		self.cv2 = tk.Canvas(self.frame2,width=self.face_frag_w, height=self.face_frag_h)
		self.cv2.pack(side=tk.TOP)

		self.face_pan = tk.Label(self.frame2, pady=8, font=(None,12))
		self.face_pan.pack(side=tk.TOP)
		self.face_roll = tk.Label(self.frame2, pady=8, font=(None,12))
		self.face_roll.pack(side=tk.TOP)
		self.face_tilt = tk.Label(self.frame2, pady=8, font=(None,12))
		self.face_tilt.pack(side=tk.TOP)

	def start(self, path, start_file, end_file):
		self.path = path
		self.current_file = start_file
		self.end_file = end_file
		self.setImage()
		self.root.mainloop()

	def nextFace(self):
		self.index += 1
		if self.index == self.face_num:
			self.current_file += 1
			if self.current_file > self.end_file:
				self.root.quit()
			else:
				self.setImage()
		else:
			self.showFace()


	def setImage(self):
		img_file = 'G00'+str(self.current_file)+'.jpg'
		self.box, self.angles = detect_faces(self.path + img_file)
		self.face_num = len(self.box)

		self.face_img = Image.open(self.path + img_file)
		self.original_img = Image.open(self.path + img_file)

		self.img_name.config(text = "Image: "+img_file)
		self.index = 0

		self.showFace()

	def showFace(self):
		i = self.index
		box_x = self.box[i][1][0] - self.box[i][0][0]
		box_y = self.box[i][2][1] - self.box[i][1][1]
		if box_x > self.face_img_w or box_y > self.face_img_h:
			scale = min(float(self.face_img_w)/float(box_x), float(self.face_img_h)/float(box_y))
		else:
			scale = max(float(self.face_img_w)/float(box_x), float(self.face_img_h)/float(box_y))
		box_x = int(box_x*scale)
		box_y = int(box_y*scale)

		boxed_img = self.original_img.copy()

		draw = ImageDraw.Draw(boxed_img)
		draw.line(self.box[i] + [self.box[i][0]], fill=128, width=10)
		del draw
		self.boxed_imgtk = ImageTk.PhotoImage(boxed_img.resize((self.ori_img_w,self.ori_img_h)))
		self.cv1.create_image((self.ori_frag_w - self.ori_img_w)/2, (self.ori_frag_h - self.ori_img_h)/2,
			image=self.boxed_imgtk, anchor='nw')

		self.cropped_imgtk = ImageTk.PhotoImage(self.face_img.copy()
			.crop(box=self.box[i][0]+self.box[i][2]).resize((box_x,box_y)))

		self.cv2.create_image((self.face_frag_w - box_x)/2, (self.face_frag_h - box_y)/2, image=self.cropped_imgtk, anchor='nw')

		self.face_index.config(text='Index: '+str(i+1)+'/'+str(self.face_num))
		self.face_pan.config(text="Face pan: "+str(self.angles[i]['pan']))
		self.face_roll.config(text="Face roll: "+str(self.angles[i]['roll']))
		self.face_tilt.config(text="Face tilt: "+str(self.angles[i]['tilt']))

def onKeyPress(event):
	pressed_char = event.char
	print('You pressed '+str(pressed_char))
	if pressed_char=='1':
		GUI.nextFace()


def labeling_gui(path, file, box, angles):
	ori_img_w = 1200
	ori_img_h = 900
	ori_frag_w = ori_img_w + 100
	ori_frag_h = ori_img_h + 100

	face_img_w = 200
	face_img_h = 300
	face_frag_w = face_img_w + 100
	face_frag_h = face_img_h + 100

	root = tk.Tk()
	root.title("Oing Labeling")
	root.bind('<KeyPress>', onKeyPress)

	frame1 = tk.Frame(root)
	frame1.pack(side=tk.LEFT)
	frame2 = tk.Frame(root)
	frame2.pack(side=tk.LEFT)

	# photo = tk.PhotoImage(Image.open(path))
	img_file = path + file
	img = Image.open(img_file)

	original_img = Image.open(img_file)
	image_total = ImageTk.PhotoImage(original_img)
	cv1 = tk.Canvas(frame1,width=ori_frag_w,height=ori_frag_h)
	cv1.create_image((ori_frag_w - ori_img_w)/2, (ori_frag_h - ori_img_h)/2, image=image_total, anchor='nw')
	cv1.pack(side=tk.TOP)

	# tk.Label(frame1,image=image_total).pack(side=tk.BOTTOM)

	img_name = tk.Label(frame2, text = "Image: "+file, padx=80, pady=12, font=(None,20, 'bold'))
	img_name.pack(side=tk.TOP)
	face_index = tk.Label(frame2, pady=8, font=(None,16,'bold'))
	face_index.pack(side=tk.TOP)

	cv = tk.Canvas(frame2,width=face_frag_w, height=face_frag_h)
	cv.pack(side=tk.TOP)

	face_pan = tk.Label(frame2, pady=8, font=(None,12))
	face_pan.pack(side=tk.TOP)
	face_roll = tk.Label(frame2, pady=8, font=(None,12))
	face_roll.pack(side=tk.TOP)
	face_tilt = tk.Label(frame2, pady=8, font=(None,12))
	face_tilt.pack(side=tk.TOP)

	images = []

	x = 0
	y = 0
	h_max = 0
	x_unit = 200
	y_unit = 300
	for i in range(len(box)):
		# box_left = min([box[i][j][0] for j in range(4)])
		# box_right = max([box[i][j][0] for j in range(4)])
		# box_upper = min([box[i][j][1] for j in range(4)])
		# box_lower = max([box[i][j][1] for j in range(4)])

		box_x = box[i][1][0] - box[i][0][0]
		box_y = box[i][2][1] - box[i][1][1]

		if box_x > face_img_w or box_y > face_img_h:
			scale = min(float(face_img_w)/float(box_x), float(face_img_h)/float(box_y))
		else:
			scale = max(float(face_img_w)/float(box_x), float(face_img_h)/float(box_y))

		box_x = int(box_x*scale)
		box_y = int(box_y*scale)

		working_img = original_img.copy()

		draw = ImageDraw.Draw(working_img)
		draw.line(box[i] + [box[i][0]], fill=128, width=10)
		del draw
		# original_img_ = working_img.resize((ori_img_w,ori_img_h))

		image_total = ImageTk.PhotoImage(working_img.resize((ori_img_w,ori_img_h)))
		# cv1 = tk.Canvas(frame1,width=ori_frag_w,height=ori_frag_h)
		cv1.create_image((ori_frag_w - ori_img_w)/2, (ori_frag_h - ori_img_h)/2, image=image_total, anchor='nw')
		# cv1.pack(side=tk.TOP)

		croped_im = img.copy().crop(box=box[i][0]+box[i][2]).resize((box_x,box_y))
		# croped_im = img.copy().crop(box=(box[i][0][0],box[i][0][1],box[i][2][0],box[i][2][1])).resize((box_x,box_y))
		image = ImageTk.PhotoImage(croped_im)
		# images.append(image)
		cv.create_image((face_frag_w - box_x)/2, (face_frag_h - box_y)/2, image=image, anchor='nw')

		face_index.config(text='Index: '+str(i+1)+'/'+str(len(box)))
		face_pan.config(text="Face pan: "+str(angles[i]['pan']))
		face_roll.config(text="Face roll: "+str(angles[i]['roll']))
		face_tilt.config(text="Face tilt: "+str(angles[i]['tilt']))
		# break

	root.mainloop()

def main():
	dir_root = 'data_gopro/'
	dir_class = 'filtered/170524_1030_101/'
	path = dir_root + dir_class
	offset = 16102
	for i in range(10):
		img_file = 'G00'+str(offset+i)+'.jpg'
		bounds, angles = detect_faces(path + img_file)
		labeling_gui(path, img_file, bounds, angles)

def main2():
	global GUI
	dir_root = 'data_gopro/'
	dir_class = 'filtered/170524_1030_101/'
	path = dir_root + dir_class
	GUI = LabelingGUI()
	GUI.start(path, 16102, 16103)
if __name__ == '__main__':
    main2()
