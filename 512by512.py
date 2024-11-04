from tkinter import *
from tkinter.ttk import Scale
from tkinter import colorchooser, filedialog, messagebox
import PIL.ImageGrab as ImageGrab
from PIL import ImageTk, Image, ImageCms, ImageFilter




################################################################

import torch
import warnings

import os
import yaml
import logging
import sys
import numpy as np
from torchvision import transforms
import argparse
import traceback
import shutil
from torchvision.utils import save_image

from runners.image_editing import Diffusion
####

from tkinter import colorchooser  ###this is what we put there

#from main import parse_args_and_config, dict2namespace, main
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():

        print(key, value)
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class Paint():
    def __init__(self, root):
        self.root = root
        self.root.title("Paint")
        self.root.geometry("1300x1020")
        self.root.configure(background="white")
        self.root.resizable(0, 0)
        


       
        self.images_ = []
        
        self.pen_color='black' 
        self.eraser_color ='white'
        
        self.color_frame = LabelFrame(self.root, text ="Color", font =('arial', 15, 'bold'), bd=5, relief=RIDGE, bg='white')
        self.color_frame.place(x=0,y=0,width=70,height=185)
        
        

        
        self.aaa=torch.ones(3, 512, 512)
        
        #print(self.aaa.shape)
        self.alpha = 1
        self.t=1000
        self.seed=3234
        
        Colors = ['#ff0000', '#ff4dd2', '#ffff33', '#000000', '#0066ff', '#660033', '#4dff4d', '#b300b3', '#00ffff', '#808080', '#99ffcc', '#336600', '#ff9966','#ff99ff', '#00cc99']
        
        
        i = j =0
        
        for color in Colors:
            Button(self.color_frame, bg=color, command=lambda col = color:self.select_color_block(col), width=3, bd=2, relief=RIDGE).grid(row=i, column=j)
            i+=1
            if i== 6: 
                i=0
                j=1
                
        self.select_color = Button(self.root, text="selecol", bd=4, bg="white",  relief=RIDGE, width=8, command=self.choose_color )
        self.select_color.place(x=0, y=187)
        
        self.clear_button = Button(self.root, text="Clear", bd=4, bg='white', command=lambda : self.canvas.delete("all"), width=8, relief=RIDGE)
        self.clear_button.place(x=0, y=217)
        
        self.save_button = Button(self.root, text="Save", bd=4, bg ='white', command=self.save_paint, width=8, relief=RIDGE)
        self.save_button.place(x=0, y=247)
       
        self.canvas_color_button = Button(self.root, text="Canvas", bd=4, bg ='white', command=self.button_color, width=8, relief=RIDGE)
        self.canvas_color_button.place(x=0, y=277)
        
        self.load_button = Button(self.root, text= "Load", bd=4, bg='white', command=self.load_image, width=8, relief=RIDGE)
        self.load_button.place(x=0, y=307)
        
        

        
        self.diffusion_button = Button(self.root, text="Diffusion", bd=4, bg='white', command=self.justdifferent, width=8, relief=RIDGE)
        self.diffusion_button.place(x=0, y=640)
        

        
       
        

       
        self.text_t = StringVar()
        self.text_timesteps= Entry(self.root, textvariable= self.text_t)
        self.text_timesteps.place(x=0, y=730)
        self.t_button = Button(self.root, text="t_steps", bd =4, bg= 'white', command=self.changetimesteps, width=8, relief=RIDGE)
        self.t_button.place(x=0, y=760)
        
        self.seed_n=StringVar()
        self.seed_steps= Entry(self.root,textvariable=self.seed_n)
        self.seed_steps.place(x=0, y=790)
        self.seed_button =  Button(self.root, text="seed", bd =4, bg= 'white', command=self.change_seed, width=8, relief=RIDGE)
        self.seed_button.place(x=0, y=820)
        
        
       
        
       
        self.run_exp_button = Button (self.root, text="Run_exp", bd= 4, bg = 'white', command = self.run_program, width=8, relief=RIDGE)
        self.run_exp_button.place(x=0, y=577)

        
        self.pen_size_scale_frame = LabelFrame(self.root, text='size', bd=5, bg='white', font=('arial', 15, 'bold'), relief=RIDGE)
        self.pen_size_scale_frame.place(x=0, y=340, height=200, width=70)
        
        self.pen_size = Scale(self.pen_size_scale_frame, orient=VERTICAL, from_ =50, to = 0, length=140  )
        
        
        self.pen_size.set(1)
        self.pen_size.grid(row=0, column=1, padx=15, )
        

        self.mask_button = Button(self.root, text="Mask", bd=4, bg= 'white', command= self.clearmask, width=8, relief=RIDGE)
        self.mask_button.place(x=0, y=607)
        
        self.whole_picture_mask_button = Button(self.root, text="Clear mask", bd=4, bg= 'white', command= self.clearwholemask, width=8, relief=RIDGE)
        self.whole_picture_mask_button.place(x=70, y=607)
        self.canvas = Canvas(self.root, bg='white', bd=0, relief=GROOVE, height=512, width=512)
        self.canvas.place(x=80, y=0)
        
        self.save_mask=Button(self.root, text="Save mask", bd=4, bg= 'white', command= self.save_mask, width=8, relief=RIDGE)
        self.save_mask.place(x=140, y=607)
        self.load_mask=Button(self.root, text="load mask", bd=4, bg= 'white', command= self.load_mask, width=8, relief=RIDGE)
        self.load_mask.place(x=210, y=607)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        



        self.canvas_2 = Canvas(self.root, bg='white', bd=0, relief=GROOVE, height=512, width=512)
        self.canvas_2.place(x=676, y=0)
        self.canvas_2.bind("<B1-Motion>", self.paint_xx)#b1-motion with self.paint
        self.canvas_2.bind("<B3-Motion>", self.unmask)#b1-motion with self.paint
        



        self.canvas_2.bind()

       

    def paint(self, event):
        
        
        x1, y1 = (event.x-2),(event.y-2)
        x2, y2 = (event.x +2),(event.y +2)
        
        
       

        self.canvas.create_oval(x1, y1, x2, y2, fill= self.pen_color, outline = self.pen_color, width = self.pen_size.get())
        self.canvas_2.create_oval(x1, y1, x2, y2, fill='#000000', outline = '#000000', width = self.pen_size.get())
        
        if x1<x2:
            xt1= x1-self.pen_size.get()
            xt1=int(xt1)
            if xt1<0:
                xt=0
            xt2=x2+ self.pen_size.get()
            #xt2=int(xt2)
            
            xt2=int(xt2)
            if xt2>512:
                xt2=512
        elif x1>x2:
            xt1 = x1 + self.pen_size.get()
            xt1=int(xt1)
            if xt1>512:
                xt1=512
            xt2= x2 - self.pen_size.get()
            xt2=int(xt2)
            if xt2<0:
                xt2=0
        if y1<y2:
            yt1= y1-self.pen_size.get()
            yt1=int(yt1)
            if yt1<0:
                yt1=0
            yt2 = y2 + self.pen_size.get()
            yt2=int(yt2)
            if yt2 > 512:
                yt2 =512
        elif y1>y2:
            yt1 = y1 + self.pen_size.get()
            yt1=int(yt1)
            if yt1>512:
                yt1=512
            yt2 = y2 - self.pen_size_get()
            yt2=int(yt2)
            if yt2<0:
                yt2=0
        
        self.aaa[:, yt1:yt2, xt1:xt2]= 0
        
    def paint_xx(self, event):
        
        print(event.x)
        print(event.y)
        x1, y1 = (event.x-2),(event.y-2)
        x2, y2 = (event.x +2),(event.y +2)
        
        
        

        self.canvas_2.create_oval(x1, y1, x2, y2, fill='#000000', outline = '#000000', width = self.pen_size.get())
        
        if x1<x2:
            xt1= x1-self.pen_size.get()
            xt1=int(xt1)
            if xt1<0:
                xt=0
            xt2=x2+ self.pen_size.get()
            #xt2=int(xt2)
            
            xt2=int(xt2)
            if xt2>512:
                xt2=512
        elif x1>x2:
            xt1 = x1 + self.pen_size.get()
            xt1=int(xt1)
            if xt1>512:
                xt1=512
            xt2= x2 - self.pen_size.get()
            xt2=int(xt2)
            if xt2<0:
                xt2=0
        if y1<y2:
            yt1= y1-self.pen_size.get()
            yt1=int(yt1)
            if yt1<0:
                yt1=0
            yt2 = y2 + self.pen_size.get()
            yt2=int(yt2)
            if yt2 > 512:
                yt2 =512
        elif y1>y2:
            yt1 = y1 + self.pen_size.get()
            yt1=int(yt1)
            if yt1>512:
                yt1=512
            yt2 = y2 - self.pen_size_get()
            yt2=int(yt2)
            if yt2<0:
                yt2=0
        
        self.aaa[:, yt1:yt2, xt1:xt2]= 0
        
    


    def unmask(self, event):
        
        x1, y1 = (event.x-2),(event.y-2)
        x2, y2 = (event.x +2),(event.y +2)
        
        
        

        
        self.canvas_2.create_oval(x1, y1, x2, y2, fill='#fff', outline = '#fff', width = self.pen_size.get())
        
        if x1<x2:
            xt1= x1-self.pen_size.get()
            xt1=int(xt1)
            if xt1<0:
                xt=0
            xt2=x2+ self.pen_size.get()
            
            
            xt2=int(xt2)
            if xt2>512:
                xt2=512
        elif x1>x2:
            xt1 = x1 + self.pen_size.get()
            xt1=int(xt1)
            if xt1>512:
                xt1=512
            xt2= x2 - self.pen_size.get()
            xt2=int(xt2)
            if xt2<0:
                xt2=0
        if y1<y2:
            yt1= y1-self.pen_size.get()
            yt1=int(yt1)
            if yt1<0:
                yt1=0
            yt2 = y2 + self.pen_size.get()
            yt2=int(yt2)
            if yt2 > 512:
                yt2 = 512
        elif y1>y2:
            yt1 = y1 + self.pen_size.get()
            yt1=int(yt1)
            if yt1>512:
                yt1=512
            yt2 = y2 - self.pen_size_get()
            yt2=int(yt2)
            if yt2<0:
                yt2=0
       
        self.aaa[:, yt1:yt2, xt1:xt2]= 1
    def select_color_block(self, col):
        self.pen_color= col
        
    def choose_color(self):
        color_code = colorchooser.askcolor(title ="Choose color")
        self.pen_color=color_code[1]
    def eraser(self):
        
        self.pen_color = eraser.color
        
    def canvas_color(self):
        color = colorchooser.askcolor()
        print(color)
        print(color[1])
        self.canvas.configure(background=color[1])
        self.eraser_color = color[1]


    def button_color(self):
        color= colorchooser.askcolor()
        self.pen_color= color[1]
    
    def save_paint(self):
        
        try:
            filename = filedialog.asksaveasfilename(defaultextension='.jpg')
            x = self.root.winfo_rootx() + self.canvas.winfo_x()
            y = self.root.winfo_rooty() + self.canvas.winfo_y()
            x1 = x +self.canvas.winfo_width()
            print(x1)
            y1 = y +self.canvas.winfo_height()
            ImageGrab.grab().crop((x+2,y+2, x1-2, y1-2)).save(filename)
            #ImageGrab.grab().crop((x,y, x1, y1)).save(filename)
            messagebox.showinfo('pain says' , 'image is saved as' +str(filename))
        except:
            messagebox.showerror("try again ")
            
    #global img        
    def load_image(self):
       
            
        filename = filedialog.askopenfilename()
        image_ = Image.open(filename)
        ph = ImageTk.PhotoImage(image_)
        self.canvas_image=ph
        my_image = self.canvas.create_image(0, 0,image=ph, anchor=NW)
        
    
   
   
        
   
    def diffusion(self):
        
        beta_start = .00001
        beta_end = .02
        num_diffusion_timesteps = 1000
        betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
        betas = torch.from_numpy(betas).float()
        num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
        (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        logvar = np.log(np.maximum(posterior_variance, 1e-20))
       
        
        filename = 'generic_save_image.jpg'#filedialog.asksaveasfilename(defaultextension='.jpg')
        
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        
        x1 = x #+self.canvas.winfo_width()
        
        y1 = y +self.canvas.winfo_height()
        ImageGrab.grab().crop((x,y, x1, y1)).save(filename)
        image = Image.open('generic_save_imagessss.jpg').convert('RGB')
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)
        image = torch.stack([image])
        
        
        img = image.to("cpu")
      
        img = img.permute(1, 2, 0, 3)
        img = img.reshape(img.shape[0], img.shape[1], -1)
        img = img / 2 + 0.5     # unnormalize
        img = torch.clamp(img, min=0., max=1.)
        npimg = img.numpy()
        
        
        x0=img
        sample_step=1000
        total_noise_levels=50
        for it in range(sample_step):
            e = torch.randn_like(x0)
            a = (1 - betas).cumprod(dim=0)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
        
        print(x.shape)
        
        save_image(x, 'your_file.jpg')
        
        
        filename= "your_file.jpg"
        image_ = Image.open(filename)
        ph = ImageTk.PhotoImage(image_)
        my_image = self.canvas.create_image(0, 0,image=ph, anchor=NW)
        self.canvas.image= ph


        mask=self.aaa

        m = {mask, image}

        torch.save(m, 'colab_demo/blurry.pth')



    def justdifferent(self):
        filename = filedialog.askopenfilename()
        image_ = Image.open(filename)
        ph = ImageTk.PhotoImage(image_)
        my_image = self.canvas.create_image(0, 0,image=ph, anchor=NW)
        self.canvas.image= ph 
       
        mask=self.aaa
         
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image_)
        m = {mask, image}
        torch.save(m, 'colab_demo/blurry.pth')



    def bigDiffusion(self):
        image_ = Image.open('generic_save_imagessss.jpg').convert('RGB')
        convert_tensor = transforms.ToTensor()
        
        image = convert_tensor(image_)
        self.image=image
        
        
        
        mask=self.aaa
        
        m = {mask, image}
        
        torch.save(m, 'colab_demo/blurry.pth')

    def clearmask(self):
        self.aaa=torch.zeros(3, 512, 512)

        ###ok to make it change it has to be torch.zeros
        self.canvas_2.create_rectangle(0, 0, 512, 512, fill='#000000', outline = '#000000', width = self.pen_size.get())


    def clearwholemask(self):
        self.aaa=torch.ones(3,512,512)
        
        self.canvas_2.delete('all')
        
        
    def save_mask(self):
       
        torch.save(self.aaa, 'savedmask.pt')
        try:
            filename = filedialog.asksaveasfilename(defaultextension='.jpg')
            print(filename)
            x = self.root.winfo_rootx() + self.canvas_2.winfo_x()
            y = self.root.winfo_rooty() + self.canvas_2.winfo_y()
            x1 = x +self.canvas_2.winfo_width()
            print(x1)
            y1 = y +self.canvas_2.winfo_height()
            ImageGrab.grab().crop((x+2,y+2, x1-2, y1-2)).save(filename)
            messagebox.showinfo('pain says' , 'image is saved as' +str(filename))
        except:
            messagebox.showerror("try again ")
    
    def load_mask(self):

       

        filename = filedialog.askopenfilename()
        image_ = Image.open(filename)
        ph = ImageTk.PhotoImage(image_)
        my_image = self.canvas_2.create_image(0, 0,image=ph, anchor=NW)
        self.canvas.image= ph 
        print(filename)
        self.aaa=torch.load('savedmask.pt')

        

    def show_mask(self):
        #here is where we will show what is going on 
        ##Perhaps we will eventually have another canvas to the side to show this.
        print('lol')
       


        
    def changetimesteps(self):


       print(self.text_timesteps.get())
       try:
            
            #if float(self.text_box.get())<=1:
            
            self.t = int(self.text_timesteps.get())
            
       except ValueError:
            print("doesnt work")


    def change_seed(self):
        try:

            self.seed= int(self.seed_steps.get())
        except ValueError:
            print('It needs to be an integer, preferably 4 digits')
        print(self.seed)
    def run_program(self):
        
        args = argparse.Namespace(comment='',  config='church.yml', exp='./runs/', image_folder='images', ni=True, npy_name='lsun_church', sample=True, sample_step=1, seed=self.seed, t=self.t, verbose='info')
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))
        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)
        
        os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
        args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder) 
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Image folder already exists. Overwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)
       # add device
        if torch.cuda.is_available():
            print("cuda is available")
        else:
            print('This will take for ever. CUDA is not available')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info("Using device: {}".format(device))
        new_config.device = device

    # set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        #print("lool this is the args.seed")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True 
        torch.cuda.is_available()
        logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Exp comment = {}".format(args.comment))
        logging.info("Config =")
        print("<" * 80)

        try:
            runner = Diffusion(args, new_config)
            runner.image_editing_sample()
        except Exception:
            logging.error(traceback.format_exc())

  




if __name__ =="__main__":
    root = Tk()
    p = Paint(root)
    root.mainloop()


