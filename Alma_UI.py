import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from AT_functions import AnimatedGIF, VideoPlayer

# Class that creates and populates a UI for LEGO ALMA
#
# Created by Adam Wikström - 2026-04-21
class ALMA_UI:
    window = None
    canvas = None
    view_occupied = False

    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("1920x1080")
        self.window.configure(bg="#000000")
        self.window.title("LEGO ALMA - Adam & Tarek")

        self.canvas = tk.Canvas(
            self.window,
            bg="#000000",
            width=1920,
            height=1080,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        self.__start()

    def __load_asset(self, path):
        base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        assets = os.path.join(base, "AT_assets")
        return os.path.join(assets, path)

    def __start(self):
        self.change_view("observation_view")
        self.window.resizable(False, False)
        self.window.mainloop()

    # Method that switches the contents of the window to select preset views
    def change_view(self, view_name):
        if self.view_occupied:
            self.canvas.delete('all')
            self.view_occupied = False

        match view_name:
            case "video_view":
                self.__video_view()
            case "observation_view":
                self.__observation_view()
            case "guide_view":
                self.__guide_view()
            case "restart_view":
                self.__restart_view()
            case "selection_view":
                self.__selection_view()

    # Method that creates the content for the video view
    def __video_view(self):
        video = VideoPlayer(self.canvas, "AT_assets\\video\\ALMA-WSU-9_2.mp4", 960, 499, size=(1703, 958))
        self.canvas.create_rectangle(262, 949, 1659, 1080,
                                fill='#eee9e9',
                                outline="#eeb005",
                                width="5.0",
                                dash=(30, 10))
        self.canvas.create_text(
            298,
            978,
            anchor="nw",
            text="Tryck på valfri knapp nedan för att starta",
            fill="#141414",
            font=("Inter", 60 * -1)
        )
        self.canvas.create_oval(1523, 961, 1628, 1066, fill="#EFB005", outline="")
        self.view_occupied = True

    # Method that creates the content for the selection view
    def __selection_view(self):
        # Galaxy
        self.canvas.create_text(
            311,
            805,
            anchor="nw",
            text="Galax",
            fill="#f2ab6d",
            font=("Inter", 32 * -1)
        )

        self.canvas.create_rectangle(234, 433, 554, 775, fill='#000000', outline="#fff9f9", width="2.0")
        img_8 = Image.open(self.__load_asset("images\\galaxy_img\\Galaxy 43.png"))
        img_8 = img_8.resize((312, 312), Image.LANCZOS)  # <- new size here
        galaxy_img = ImageTk.PhotoImage(img_8)
        self.canvas.create_image(394, 604, image=galaxy_img)
        self.canvas.galaxy_img = galaxy_img  # keep reference

        # Protoplanetary Disc
        self.canvas.create_text(
            632,
            796,
            anchor="nw",
            text="Protoplanetär Skiva",
            fill="#f2ab6d",
            font=("Inter", 32 * -1)
        )

        self.canvas.create_rectangle(609, 433, 929, 775, fill='#000000', outline="#fff9f9", width="2.0")
        img_9 = Image.open(self.__load_asset("images\\disc_img\\Disk_43.png"))
        img_9 = img_9.resize((312, 312), Image.LANCZOS)  # <- new size here
        disc_img = ImageTk.PhotoImage(img_9)
        self.canvas.create_image(769, 604, image=disc_img)
        self.canvas.disc_img = disc_img

        self.canvas.create_text(
            1072,
            796,
            anchor="nw",
            text="Stjärna",
            fill="#f2ab6d",
            font=("Inter", 32 * -1)
        )

        self.canvas.create_rectangle(983, 433, 1303, 775, fill='#000000', outline="#fff9f9", width="2.0")
        img_10 = Image.open(self.__load_asset("images\\star_img\\star_43.png"))
        img_10 = img_10.resize((312, 312), Image.LANCZOS)  # <- new size here
        star_img = ImageTk.PhotoImage(img_10)
        self.canvas.create_image(1143, 604, image=star_img)
        self.canvas.star_img = star_img

        self.canvas.create_text(
            1346,
            796,
            anchor="nw",
            text="Jetstråle",
            fill="#f2ab6d",
            font=("Inter", 32 * -1)
        )

        self.canvas.create_rectangle(1357, 433, 1677, 775, fill='#000000', outline="#fff9f9", width="2.0")
        img_11 = Image.open(self.__load_asset("images\\jet_img\\Jet_43.png"))
        img_11 = img_11.resize((312, 312), Image.LANCZOS)  # <- new size here
        jet_img = ImageTk.PhotoImage(img_11)
        self.canvas.create_image(1517, 604, image=jet_img)
        self.canvas.jet_img = jet_img

        self.canvas.create_text(
            240,
            152,
            anchor="nw",
            text="Välj vad ni vill se via knapparna \ntill vänster",
            fill="#f2ab6d",
            font=("Inter", 96 * -1),
        )

        arrow_pos = [105, 226, 339, 463]
        arrow_images = []
        for i in range(4):
            temp_image = Image.open(self.__load_asset("gifs\\Arrow_blink.gif"))
            temp_image = temp_image.convert("RGBA")
            temp_image = temp_image.resize((184, 130), Image.LANCZOS)  # <- new size here
            temp_image = temp_image.rotate(76, expand=True)
            arrow_img = ImageTk.PhotoImage(temp_image)
            self.canvas.create_image(arrow_pos[i], 975, image=arrow_img)
            arrow_images.append(arrow_img)

        self.view_occupied = True

    # Method that creates the content for the guide view
    def __guide_view(self):
        self.canvas.create_text(
            260,
            122,
            anchor="nw",
            text="Sätt antenner på de vita brickorna",
            fill="#f2ab6d",
            font=("Inter", 96 * -1)
        )

        AnimatedGIF(self.canvas,
                    "AT_assets\\gifs\\placing-antennas.gif",
                    959, 729,
                    size=(1279, 904))  # Added 2026-04-20 - Adam W
        self.view_occupied = True

    # Method that creates the content for the observation view
    def __observation_view(self):
        #
        self.canvas.create_rectangle(269, 63, 528, 324, fill='#000000', outline="#ffffff", width="3.0")
        # Information/Guide GIF rectangle
        self.canvas.create_rectangle(156, 670, 641, 1013, fill='#000000', outline="#ffffff", width="3.0")
        # Observation Rectangle
        self.canvas.create_rectangle(794, 63, 1744, 1013, fill='#141414', outline="#ffffff", width="3.0")

        place_gif = AnimatedGIF(self.canvas,
                    "AT_assets\\gifs\\placing-antennas.gif",
                    398, 843,
                    size=(449, 343))

        # spread_gather_gif = AnimatedGIF(self.canvas,
        #                                 "AT_assets\\gifs\\spread_and_gather_antennas-updated.gif",
        #                                 415, 843,
        #                                 size=(449, 343))
        # spread_gather_gif.set_speed(1.3)

        # image_5 = tk.PhotoImage(file=load_asset("None"))

        # canvas.create_image(398, 842, image=image_5)

        # image_6 = tk.PhotoImage(file=load_asset("None"))

        # canvas.create_image(398, 843, image=image_6)

        image_7 = Image.open(self.__load_asset("images\\antenna icon 1.png"))
        image_7 = image_7.resize((56, 66), Image.LANCZOS)  # <- new size here
        antenna_img = ImageTk.PhotoImage(image_7)
        self.canvas.create_image(1073, 949, image=antenna_img)

        label_1 = tk.Label(
            text="42/42",
            fg="#ffffff",
            bg="#151414",
            font=("Inter", 40 * -1),
            anchor="e"
        )

        label_1.place(x=920, y=930)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor="#151414",  # match window bg
                        background="#00EE00",  # progress fill color
                        bordercolor="white",
                        lightcolor="green",
                        darkcolor="green")
        progress_bar = ttk.Progressbar(self.window,
                                       style="Custom.Horizontal.TProgressbar",
                                       orient="horizontal",
                                       length=370,
                                       mode="determinate")

        progress_bar.pack(pady=20)
        progress_bar.pack(pady=20)
        progress_bar["maximum"] = 100
        progress_bar["value"] = 100
        progress_bar.place(x=1155, y=950)

        self.view_occupied = True

    # Method that creates the content for the restart view
    def __restart_view(self):
        self.canvas.create_text(
            357,
            65,
            anchor="nw",
            text="   Flytta ALLA antenner till \ninhägnaden för att börja om",
            fill="#f2ab6d",
            font=("Inter", 96 * -1)
        )

        AnimatedGIF(self.canvas,
                    "AT_assets\\gifs\\remove-antennas.gif",
                    959, 644,
                    size=(1455, 1028))  # Added 2026-04-20 - Adam W
        self.view_occupied = True

    def get_canvas(self):
        return self.canvas
