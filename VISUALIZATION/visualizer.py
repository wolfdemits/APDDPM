import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib
from PIL import Image

import threading

import tkinter
import customtkinter
from matplotlib.backends.backend_agg import FigureCanvasAgg
matplotlib.use('agg')

from PREPROCESSING.datamanager import Datamanager

DATAPATH = pathlib.Path('./DATA')

datamanager = Datamanager(DATAPATH)

## interface ##
root = customtkinter.CTk()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# global variable init
scan = None
divisions = None
worker_thread = None
available_slices_1 = 0
available_slices_2 = 0
slice_idx_1 = None
slice_idx_2 = None
view1_div = 0
view2_div = 0

root.geometry("750x600")
w = 1000
h = 800
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.title("PET scan Visualizer")
root.resizable(False, False)

img_width = 400
img_height = 400

frame = customtkinter.CTkFrame(master=root)
frame.pack(side="top", fill="both", expand=True)

## Rendering Logic #######################################################

## Render function for matplotlib plots
def fig_to_pil(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    
    # Get ARGB data and convert to RGB
    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    argb = argb.reshape((height, width, 4))  # ARGB format (A, R, G, B)
    
    # Swap ARGB → RGB (drop alpha)
    rgb = argb[:, :, 1:]  # Remove alpha channel
    
    image = Image.fromarray(rgb, mode="RGB")
    return image

## Refresh function to be called whenever figures change
def refresh_screen():
    # current patient id
    patientid_label.configure(text=datamanager.current_patient)

    global slice_idx_1, slice_idx_2
    slice_idx_1 = None
    slice_idx_2 = None

    # load 
    def load_scan():
        plt.close() # close previous plot instances
        return datamanager.load_scan()
    
    def start_task():
        global worker_thread, result_var
        result_var = None
        root.configure(cursor="watch")  # show busy cursor

        def task_wrapper():
            global scan, divisions
            scan, divisions = load_scan()

        worker_thread = threading.Thread(target=task_wrapper, daemon=True)
        worker_thread.start()
        root.after(100, check_done)  # poll every 100ms

    def check_done():
        if worker_thread.is_alive():
            root.after(100, check_done)
        else:
            root.configure(cursor="")  # reset cursor

            soft_refresh()

    start_task()

def soft_refresh():
    global view1_ori
    global view2_ori
    global scan, divisions
    global available_slices_1, available_slices_2
    global slice_idx_1, slice_idx_2
    global view1_div, view2_div

    
    # update slider and division info
    slider1.configure(to=0, from_=(len(divisions) - 1), number_of_steps=len(divisions))
    slider2.configure(to=0, from_=(len(divisions) - 1), number_of_steps=len(divisions))

    if view1_ori.get() == 'Transaxial':
        available_slices_1 = scan[view1_div].shape[0]
        if slice_idx_1 is None:
            slice_idx_1 = available_slices_1 // 2
        img1_arr = scan[view1_div][slice_idx_1]
    elif view1_ori.get() == 'Saggital':
        available_slices_1 = scan[view1_div].shape[1]
        if slice_idx_1 is None:
            slice_idx_1 = available_slices_1 // 2
        img1_arr = scan[view1_div][:, slice_idx_1]
    else:
        available_slices_1 = scan[view1_div].shape[2]
        if slice_idx_1 is None:
            slice_idx_1 = available_slices_1 // 2
        img1_arr = scan[view1_div][:, :, slice_idx_1]

    if view2_ori.get() == 'Transaxial':
        available_slices_2 = scan[view2_div].shape[0]
        if slice_idx_2 is None:
            slice_idx_2 = available_slices_2 // 2
        img2_arr = scan[view2_div][slice_idx_2]
    elif view2_ori.get() == 'Saggital':
        available_slices_2 = scan[view2_div].shape[1]
        if slice_idx_2 is None:
            slice_idx_2 = available_slices_2 // 2
        img2_arr = scan[view2_div][:, slice_idx_2]
    else:
        available_slices_2 = scan[view2_div].shape[2]
        if slice_idx_2 is None:
            slice_idx_2 = available_slices_2 // 2
        img2_arr = scan[view2_div][:, :, slice_idx_2]

    # set slice info
    slice_1_label.configure(text=f'{slice_idx_1}/{available_slices_1 - 1}')
    slice_2_label.configure(text=f'{slice_idx_2}/{available_slices_2 - 1}')

    #### image rendering ###############
    fig_img1, ax_img1 = plt.subplots(figsize=(5, 5))
    ax_img1.imshow(img1_arr) #, cmap="gray")
    fig_img1.patch.set_facecolor('#242424ff')
    ax_img1.axis("off")
    fig_img1.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig_img2, ax_img2 = plt.subplots(figsize=(5, 5))
    ax_img2.imshow(img2_arr) #, cmap="gray")
    fig_img2.patch.set_facecolor('#242424ff')
    ax_img2.axis("off")
    fig_img2.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert Matplotlib figure to a PIL image
    pil_img1 = fig_to_pil(fig_img1)
    pil_img2 = fig_to_pil(fig_img2)

    # update ctk image
    img1_ctk = customtkinter.CTkImage(light_image=pil_img1, size=(img_width,img_height))
    img1_label.configure(image=img1_ctk)

    img2_ctk = customtkinter.CTkImage(light_image=pil_img2, size=(img_width,img_height))
    img2_label.configure(image=img2_ctk)

    # Close the figures to free memory
    plt.close(fig_img1)
    plt.close(fig_img2)

    ####################################

    return

##########################################################################

## Element Logic #########################################################

# next/previous patient logic
def next_patient():
    datamanager.next_patient()
    refresh_screen()

def previous_patient():
    datamanager.previous_patient()
    refresh_screen()

# next/previous slice logic
def next_slice_1():
    global slice_idx_1, available_slices_1
    if slice_idx_1 >= (available_slices_1 - 1):
        slice_idx_1 = 0
    else:
        slice_idx_1 += 1

    soft_refresh()

def next10_slice_1():
    global slice_idx_1, available_slices_1
    if slice_idx_1 + 9 >= (available_slices_1 - 1):
        slice_idx_1 = 0
    else:
        slice_idx_1 += 10

    soft_refresh()

def previous_slice_1():
    global slice_idx_1, available_slices_1
    if slice_idx_1 <= 0:
        slice_idx_1 = available_slices_1 - 1
    else:
        slice_idx_1 -= 1

    soft_refresh()
    
def previous10_slice_1():
    global slice_idx_1, available_slices_1
    if slice_idx_1 - 10 <= 0:
        slice_idx_1 = available_slices_1 - 1
    else:
        slice_idx_1 -= 10

    soft_refresh()

def next_slice_2():
    global slice_idx_2, available_slices_2
    if slice_idx_2 >= (available_slices_2 - 1):
        slice_idx_2 = 0
    else:
        slice_idx_2 += 1

    soft_refresh()

def next10_slice_2():
    global slice_idx_2, available_slices_2
    if slice_idx_2 + 9 >= (available_slices_2 - 1):
        slice_idx_2 = 0
    else:
        slice_idx_2 += 10

    soft_refresh()

def previous_slice_2():
    global slice_idx_2, available_slices_2
    if slice_idx_2 <= 0:
        slice_idx_2 = available_slices_2 - 1
    else:
        slice_idx_2 -= 1

    soft_refresh()

def previous10_slice_2():
    global slice_idx_2, available_slices_2
    if slice_idx_2 - 10 <= 0:
        slice_idx_2 = available_slices_2 - 1
    else:
        slice_idx_2 -= 10

    soft_refresh()

# dropdown logic
def dropdown_callback(choice):
    # set tracer
    datamanager.set_tracer(choice)
    refresh_screen()

def view1_ori_callback(choice):
    global slice_idx_1
    slice_idx_1 = None
    soft_refresh()

def view2_ori_callback(choice):
    global slice_idx_2
    slice_idx_2 = None
    soft_refresh()

# div slider logic
def slider1_callback(value):
    int_value = int(round(value))
    slider1.set(int_value)

    if divisions is None:
        return
    
    global view1_div
    view1_div = int_value
    slider1_label_value.configure(text=f'div{divisions[view1_div]}')
    soft_refresh()

def slider2_callback(value):
    int_value = int(round(value))
    slider2.set(int_value)

    if divisions is None:
        return
    
    global view2_div
    view2_div = int_value
    slider2_label_value.configure(text=f'div{divisions[view2_div]}')
    soft_refresh()

## GUI Element definitions ###############################################

# Tracer dropdown
tracer_label = customtkinter.CTkLabel(text="Tracer Type", master=root, text_color='#c92477', font=(None,18))
tracer_label.place(relx = 0.04, rely=0.02)

tracer = customtkinter.StringVar(value=datamanager.available_tracers[0])
tracermenu = customtkinter.CTkComboBox(root, values=datamanager.available_tracers, command=dropdown_callback, variable=tracer)
tracermenu.place(relx=0.04, rely=0.07, relwidth=0.2)

# Patient selector
patient_label = customtkinter.CTkLabel(text="Patient", master=root, text_color='#c92477', font=(None,18))
patient_label.place(relx = 0.675, rely=0.02)

next_patient_button = customtkinter.CTkButton(text=">", master=root, corner_radius=5, command=lambda: next_patient())
next_patient_button.place(relx=0.92, rely=0.08, relwidth=0.05, anchor=tkinter.CENTER)
patientid_label = customtkinter.CTkLabel(text="Patient ID", master=root, text_color="#366abf", font=(None,16))
patientid_label.place(relx = 0.8, rely=0.06)
previous_patient_button = customtkinter.CTkButton(text="<", master=root, corner_radius=5, command=lambda: previous_patient())
previous_patient_button.place(relx=0.70, rely=0.08, relwidth=0.05, anchor=tkinter.CENTER)

# view dropdown
view1_ori = customtkinter.StringVar(value='Transaxial')
view1_ori_menu = customtkinter.CTkComboBox(root, values=['Transaxial', 'Saggital', 'Coronal'], command=view1_ori_callback, variable=view1_ori)
view1_ori_menu.place(relx = 0.18, rely=0.13, relwidth=0.20)

view2_ori = customtkinter.StringVar(value='Transaxial')
view2_ori_menu = customtkinter.CTkComboBox(root, values=['Transaxial', 'Saggital', 'Coronal'], command=view2_ori_callback, variable=view2_ori)
view2_ori_menu.place(relx = 0.62, rely=0.13, relwidth=0.20)

# images
img1_ctk = customtkinter.CTkImage(light_image=Image.new("RGB", (img_width, img_height)), size=(img_width, img_height)) # placeholder image
img1_label = customtkinter.CTkLabel(root, image=img1_ctk, text="")
img1_label.place(relx=0.09, rely=0.18)
img2_ctk = customtkinter.CTkImage(light_image=Image.new("RGB", (img_width, img_height)), size=(img_width, img_height)) # placeholder image
img2_label = customtkinter.CTkLabel(root, image=img2_ctk, text="")
img2_label.place(relx=0.52, rely=0.18)

# slice selector
next10_slice_1_button = customtkinter.CTkButton(text=">>", master=root, corner_radius=5, command=lambda: next10_slice_1())
next10_slice_1_button.place(relx=0.45, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)
next_slice_1_button = customtkinter.CTkButton(text=">", master=root, corner_radius=5, command=lambda: next_slice_1())
next_slice_1_button.place(relx=0.38, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)
slice_1_label = customtkinter.CTkLabel(text="", master=root, text_color="#5ab098", font=(None,16))
slice_1_label.place(relx = 0.27, rely=0.69)
previous_slice_1_button = customtkinter.CTkButton(text="<", master=root, corner_radius=5, command=lambda: previous_slice_1())
previous_slice_1_button.place(relx=0.21, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)
previous10_slice_1_button = customtkinter.CTkButton(text="<<", master=root, corner_radius=5, command=lambda: previous10_slice_1())
previous10_slice_1_button.place(relx=0.13, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)

next10_slice_2_button = customtkinter.CTkButton(text=">>", master=root, corner_radius=5, command=lambda: next10_slice_2())
next10_slice_2_button.place(relx=0.89, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)
next_slice_2_button = customtkinter.CTkButton(text=">", master=root, corner_radius=5, command=lambda: next_slice_2())
next_slice_2_button.place(relx=0.82, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)
slice_2_label = customtkinter.CTkLabel(text="", master=root, text_color="#5ab098", font=(None,16))
slice_2_label.place(relx = 0.69, rely=0.69)
previous_slice_2_button = customtkinter.CTkButton(text="<", master=root, corner_radius=5, command=lambda: previous_slice_2())
previous_slice_2_button.place(relx=0.63, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)
previous10_slice_2_button = customtkinter.CTkButton(text="<<", master=root, corner_radius=5, command=lambda: previous10_slice_2())
previous10_slice_2_button.place(relx=0.55, rely=0.71, relwidth=0.05, anchor=tkinter.CENTER)

# division slider
slider1_label = customtkinter.CTkLabel(text="DIVISION", master=root, text_color="#366abf", font=(None,16))
slider1_label.place(relx = 0.008, rely=0.17)
slider1 = customtkinter.CTkSlider(master=root, from_=6, to=0, command=slider1_callback, number_of_steps=7, orientation='vertical')
slider1.place(relx=0.04, rely=0.21, relheight=0.32)
slider1.set(0)
slider1_label_value = customtkinter.CTkLabel(text='', master=root, fg_color="transparent", text_color="#5ab098", font=(None,16))
slider1_label_value.place(relx=0.03, rely=0.54)

slider2_label = customtkinter.CTkLabel(text="DIVISION", master=root, text_color="#366abf", font=(None,16))
slider2_label.place(relx = 0.925, rely=0.17)
slider2 = customtkinter.CTkSlider(master=root, from_=6, to=0, command=slider2_callback, number_of_steps=7, orientation='vertical')
slider2.place(relx=0.95, rely=0.21, relheight=0.32)
slider2.set(0)
slider2_label_value = customtkinter.CTkLabel(text='', master=root, fg_color="transparent", text_color="#5ab098", font=(None,16))
slider2_label_value.place(relx=0.93, rely=0.54)

# info labels

##########################################################################

refresh_screen()
root.mainloop()