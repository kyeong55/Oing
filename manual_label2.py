# import io
import csv 
import sys
import os
import time

import tkinter as tk
from PIL import ImageTk, Image, ImageDraw

FACE_DETECTION_MAX_FACES = 32
GUI = None

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
        
    def start(self, path, csv_file_read, csv_file_write):
        self.index = 0
        self.img_file = ""
        self.path = path
        self.csv_file_write = csv_file_write
        with open(csv_file_read) as f:
            csv_reader = csv.reader(f, delimiter=',', lineterminator='\n')
            csv_reader.__next__()
            self.detection = []
            self.data = []
            for row in csv_reader:
                self.detection.append(row)
            self.face_num = len(self.detection)
        if os.path.isfile(csv_file_write):
            with open(csv_file_write) as prev:
                reader = csv.reader(prev, delimiter=',', lineterminator='\n')
                next(reader)
                for row in reader:
                    self.data.append(row)
                    self.index +=1
        
        self.showFace()
        self.root.mainloop()
        # with open(csv_file_write,'w') as out:
        #     self.csv_writer = csv.writer(out, delimiter=',', lineterminator='\n')
        #     fieldnames=['dir_name','img_file','bound_left','bound_top','bound_right','bound_bot','pan','roll','tilt','anger','joy','sorrow','surprise','label']
        #     self.csv_writer.writerow(fieldnames)
            

    def saveData(self):
        with open(self.csv_file_write,'w') as out:
            csv_writer = csv.writer(out, delimiter=',', lineterminator='\n')
            fieldnames=['dir_name','img_file','bound_left','bound_top','bound_right','bound_bot','pan','roll','tilt','anger','joy','sorrow','surprise','label']
            csv_writer.writerow(fieldnames)
            csv_writer.writerows(self.data)

    def prevFace(self):
        if self.index == 0:
            return
        self.index -= 1
        self.showFace()

    def nextFace(self):
        if self.index >= self.face_num - 1:
            self.root.quit()
        self.index += 1
        self.showFace()

    def showFace(self):
        row = self.detection[self.index]
        if self.img_file != row[1]:
            self.img_file = row[1]
            self.img_name.config(text = "Image: "+self.img_file)
            self.face_img = Image.open(self.path + row[0] + self.img_file)
            self.original_img = Image.open(self.path + row[0] + self.img_file)

        bound_l = int(row[2])
        bound_t = int(row[3])
        bound_r = int(row[4])
        bound_b = int(row[5])
        bound = [(bound_l,bound_t),(bound_r,bound_t),(bound_r,bound_b),(bound_l,bound_b),(bound_l,bound_t)]
        box_x = bound_r - bound_l
        box_y = bound_b - bound_t
        
        if box_x > self.face_img_w or box_y > self.face_img_h:
            scale = min(float(self.face_img_w)/float(box_x), float(self.face_img_h)/float(box_y))
        else:
            scale = max(float(self.face_img_w)/float(box_x), float(self.face_img_h)/float(box_y))
        box_x = int(box_x*scale)
        box_y = int(box_y*scale)

        boxed_img = self.original_img.copy()

        draw = ImageDraw.Draw(boxed_img)
        draw.line(bound, fill=(0,255,0,255), width=10)
        del draw
        self.boxed_imgtk = ImageTk.PhotoImage(boxed_img.resize((self.ori_img_w,self.ori_img_h)))
        self.cv1.create_image((self.ori_frag_w - self.ori_img_w)/2, (self.ori_frag_h - self.ori_img_h)/2,
            image=self.boxed_imgtk, anchor='nw')

        self.cropped_imgtk = ImageTk.PhotoImage(self.face_img.copy()
            .crop(box=[bound_l,bound_t,bound_r,bound_b]).resize((box_x,box_y)))

        self.cv2.create_image((self.face_frag_w - box_x)/2, (self.face_frag_h - box_y)/2, image=self.cropped_imgtk, anchor='nw')

        self.face_index.config(text='Index: '+str(self.index+1)+'/'+str(self.face_num))
        self.face_pan.config(text="Face pan: "+row[6])
        self.face_roll.config(text="Face roll: "+row[7])
        self.face_tilt.config(text="Face tilt: "+row[8])

def onKeyPress(event):
    pressed_char = event.char
    print('You pressed '+str(pressed_char))
    if pressed_char == '1':
        # TODO : db write 
        GUI.data.append(GUI.detection[GUI.index] + [1])
        # GUI.csv_writer.writerow(GUI.row + [1])
        GUI.nextFace()
    if pressed_char == '2':
        # TODO : db write 
        GUI.data.append(GUI.detection[GUI.index] + [2])
        GUI.nextFace()
    if pressed_char == '3':
        # TODO : db write 
        GUI.data.append(GUI.detection[GUI.index] + [3])
        GUI.nextFace()
    if pressed_char == 'q':
        # TODO : quit
        GUI.root.quit()
    if pressed_char == 'z':
        GUI.data.pop(len(GUI.data)-1)
        GUI.prevFace()

def main():
    start_time = time.time()
    global GUI
    path = 'data_gopro/filtered/'
    csv_file_read = 'detection_0900.csv'
    csv_file_write = 'detection_0900_labeled.csv'
    GUI = LabelingGUI()
    GUI.start(path,csv_file_read,csv_file_write)
    GUI.saveData()
    s = int(time.time() - start_time)
    print('The time you lost: {:02}:{:02}:{:02}'.format(s//3600, s%3600//60, s%60))

if __name__ == '__main__':
    main()
