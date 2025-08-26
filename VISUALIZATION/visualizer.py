import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import json
from PIL import Image

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

frame = customtkinter.CTkFrame(master=root)
frame.pack(side="top", fill="both", expand=True)
## 

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

    # load 
    plt.close() # close previous plot instances
    scan, divisions = datamanager.load_scan()

    ### TODO
    # DEBUG, for now
    view1_div = 0
    view2_div = 0

    # DEBUG, for now
    slice_idx_1 = 100
    slice_idx_2 = 200

    img1_arr = scan[view1_div][slice_idx_1]
    img2_arr = scan[view2_div][:,slice_idx_2]

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
    img1_ctk = customtkinter.CTkImage(light_image=pil_img1, size=(300,300))
    img1_label.configure(image=img1_ctk)

    img2_ctk = customtkinter.CTkImage(light_image=pil_img2, size=(300,300))
    img2_label.configure(image=img2_ctk)

    # Close the figures to free memory
    plt.close(fig_img1)
    plt.close(fig_img2)

    ####################################

    return

# next/previous patient logic
def next_patient():
    datamanager.next_patient()
    refresh_screen()

def previous_patient():
    datamanager.previous_patient()
    refresh_screen()

# dropdown logic
def dropdown_callback(choice):
    # set tracer
    datamanager.set_tracer(choice)
    refresh_screen()

## GUI Element definitions ##

# Tracer dropdown
tracer_label = customtkinter.CTkLabel(text="Tracer Type", master=root, text_color='#c92477', font=(None,18))
tracer_label.place(relx = 0.04, rely=0.02)

tracer = customtkinter.StringVar(value=datamanager.available_tracers[0])
combobox = customtkinter.CTkComboBox(root, values=datamanager.available_tracers, command=dropdown_callback, variable=tracer)
combobox.place(relx=0.04, rely=0.07, relwidth=0.2)

# Patient selector
patient_label = customtkinter.CTkLabel(text="Patient", master=root, text_color='#c92477', font=(None,18))
patient_label.place(relx = 0.68, rely=0.02)

next_patient_button = customtkinter.CTkButton(text=">", master=root, corner_radius=5, command=lambda: next_patient())
next_patient_button.place(relx=0.92, rely=0.08, relwidth=0.05, anchor=tkinter.CENTER)
patientid_label = customtkinter.CTkLabel(text="Patient ID", master=root, text_color="#366abf", font=(None,16))
patientid_label.place(relx = 0.8, rely=0.06)
previous_patient_button = customtkinter.CTkButton(text="<", master=root, corner_radius=5, command=lambda: previous_patient())
previous_patient_button.place(relx=0.70, rely=0.08, relwidth=0.05, anchor=tkinter.CENTER)

# images
img1_label = customtkinter.CTkLabel(text="View 1", master=root, text_color="#5ab098", font=(None,18))
img1_label.place(relx = 0.14, rely=0.23, relwidth=0.25)
img2_label = customtkinter.CTkLabel(text="View 2", master=root, text_color="#5ab098", font=(None,18))
img2_label.place(relx = 0.62, rely=0.23, relwidth=0.25)

img1_ctk = customtkinter.CTkImage(light_image=Image.new("RGB", (500, 500)), size=(500,500)) # placeholder image
img1_label = customtkinter.CTkLabel(root, image=img1_ctk, text="")
img1_label.place(relx=0.06, rely=0.28)
img2_ctk = customtkinter.CTkImage(light_image=Image.new("RGB", (500, 500)), size=(500,500)) # placeholder image
img2_label = customtkinter.CTkLabel(root, image=img2_ctk, text="")
img2_label.place(relx=0.54, rely=0.28)

refresh_screen()
root.mainloop()


#### TODO
# - add view + slice selection
# - add division sliders
# - add info view
# - add imshow vmax/vmin functionality!!!