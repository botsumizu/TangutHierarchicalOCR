import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
from thocr_system import THOCRSystem


thocr = THOCRSystem()

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("image file", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    img_resized = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_resized)
    img_label.config(image=img_tk)
    img_label.image = img_tk


    result = thocr.predict(file_path)
    if "error" in result:
        result_text.set(f"{result['error']}")
    else:
        structure = result["structure"]
        character = result["character"]
        conf_s = result["confidence"]["structure"]
        conf_c = result["confidence"]["character"]

        result_text.set(
            f"structure: {structure}（{conf_s:.2%}）\n"
            f"character: {character}（{conf_c:.2%}）"
        )

root = tk.Tk()
root.title("THOCR")
root.geometry("400x500")

btn = tk.Button(root, text="select a image to recognize", command=select_image, height=2)
btn.pack(pady=20)

img_label = Label(root)
img_label.pack()

result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Arial", 14), justify="left")
result_label.pack(pady=20)

root.mainloop()
