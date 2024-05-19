from mltu.configs import BaseModelConfigs
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from interfaceModel import ImageToWordModel


# model_name = "2024-05-12--1844" # low performance, cer=17%
model_name="202301111911" # high performance, cer=7%
if(len(model_name) == 0):
    raise SystemExit("No model selected.")
model_path = f"Model/{model_name}/configs.yaml"
configs = BaseModelConfigs.load(model_path)
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Handwriting OCR App")
        self.geometry("400x560")

        # Create two frames for image and input
        self.image_frame = ctk.CTkFrame(self, width=400, height=360)
        self.image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew",)
        self.grid_rowconfigure(1, weight=2)
        self.grid_columnconfigure(0, weight=1)
        

        self.control_frame = ctk.CTkFrame(self, width=400, height=200)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create widgets for the image frame
        self.image_label = ctk.CTkLabel(self.image_frame, text="Image Preview")
        # self.image_label.pack(pady=10)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        self.open_file_button = ctk.CTkButton(self.control_frame, text="Select image", command=self.open_file_dialog)
        self.open_file_button.pack(pady=10)

        self.recognized_text_label = ctk.CTkLabel(self.control_frame, text="Recognized text : ", wraplength=350)
        self.recognized_text_label.pack(pady=10)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            aspect_ratio = image.shape[1] / image.shape[0]
            if aspect_ratio > 1:
                image = cv2.resize(image, (360, int(360 / aspect_ratio)))
            else:
                image = cv2.resize(image, (int(360 * aspect_ratio), 360))
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            self.image_label.configure(image=photo,text="")
            # this line is necessary to prevent garbage collection
            self.image_label.image = photo

            recognized_text = model.predict(image)

            # recognized_text = "placeholder text"
            self.recognized_text_label.configure(text=f"Recognized text : {recognized_text}")

if __name__ == "__main__":
    app = App()
    app.mainloop()