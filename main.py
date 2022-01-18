from carDetection import *
from LPDetection import *
import time

NS_TO_MS = 1/1000000

def updateImage (image):
    global currentImage
    # The window is resized 
    h,w = np.size(image,0),np.size(image,1)
    resizeWindow(w,h+30)
    resizeFrame(w,h)
    # And the image is updated
    currentImage = image
    B,G,R = cv2.split(image)
    imgMerged = cv2.merge((R,G,B))
    img = Image.fromarray(imgMerged)
    tkimg = ImageTk.PhotoImage(img)
    imgLabel.config(image=tkimg)
    imgLabel.tkimg = tkimg
    imgLabel.pack()

def resizeWindow(w,h):
    ui.geometry(f"{w}x{h}")

def resizeFrame(w,h):
    frame.config(width = w)
    frame.config(height = h)

def browseFiles():
    global carDetecter
    # The path of the file is saved
    filePath = filedialog.askopenfilename(initialdir = "/",title = "Select a File",
                                          filetypes = (('image', '*.jpg'),('image', '*.jpeg')))
    # The image is load in the car detecter
    if len(filePath):
        img = cv2.imread(filePath)
        carDetecter.setImage(img)
        updateImage(img)
        carTXT.config(text = "Nº Cars = -")
        timeTXT.config(text = "Detection time = -")
        fileMenu.entryconfig("Save As", state="normal")

def saveFile():
    global carDetecter
    global currentImage
    # The path for saving the file is saved
    saveFilePath = filedialog.asksaveasfile(mode='w', initialdir = "/", title = "Save File as",
                                            filetypes = [('image', '*.jpg')])
    # The file is saved
    if not saveFilePath:
        return
    cv2.imwrite(saveFilePath.name+".jpg",currentImage)

def LPdetection ():
    global carDetecter 
    # The detection time is temporized
    t0 = time.time_ns()
    imgResult = carDetecter.detection()
    t = (time.time_ns() - t0) * NS_TO_MS
    # And shown in the GUI
    timeTXT.config(text = "Detection time = %.2fms" % t)
    carTXT.config(text = "Nº Cars = "+str(len(carDetecter.validOutputBoxes)))
    updateImage(imgResult)

def defaultPanel():
    # The default image is shown
    baseimg = ImageTk.PhotoImage(Image.open("byDefault.jpg"))
    imgLabel.config(image=baseimg)
    imgLabel.tkimg = baseimg 
    imgLabel.pack()

def LPDetectionOpt(option):
    carDetecter.LPDetector.detectionType = option
    # The correspondent option is marked as active in the menu, changing the background color
    if option == "Yolo":
        LPDetectionMenu.entryconfig("Yolo", background = "gray82")
        LPDetectionMenu.entryconfig("Edge detection", background = "#F0F0ED")
    else:
        LPDetectionMenu.entryconfig("Yolo", background = "#F0F0ED")
        LPDetectionMenu.entryconfig("Edge detection", background = "gray82")

def CarDetectionOpt(option):
    carDetecter.detectionType = option
    carDetecter.loadModel()
    # The correspondent option is marked as active in the menu, changing the background color
    if option == "Yolo":
        carDetectionMenu.entryconfig("Yolo", background = "gray82")
        carDetectionMenu.entryconfig("Yolo Tiny", background = "#F0F0ED")
    else:
        carDetectionMenu.entryconfig("Yolo", background = "#F0F0ED")
        carDetectionMenu.entryconfig("Yolo Tiny", background = "gray82")

if __name__ == "__main__":
    carDetecter = CarDetector(modelCfg='yolov3.cfg', modelWeights='yolov3.weights', threshold=0.7)
    LPDetecterOption = "Yolo"

    # The main window is created, with its characteristics
    ui = Tk()
    ui.title("LP Detection - FutureVision")
    ui.geometry("640x530")

    # Setting icon of master window
    logo = PhotoImage(file = 'logo.png')
    ui.iconphoto(False, logo)
    menuBar = Menu(ui)

    # The frame for displaying the image is created
    frame = Frame(ui, width=640, height=480)
    frame.pack()
    frame.pack_propagate(0)
    frame.grid(row=0, column=0, columnspan=3, sticky="nswe")
    imgLabel = Label(frame)
    defaultPanel()

    # The detection button is created
    button = Button(text="Start detection", command=LPdetection)
    button.grid(row=1, column=1)

    # A text with the number os cars is shown
    carTXT = Label(ui, text = "Nº Cars = -")
    carTXT.grid(row=1, column=0)

    # A text with the number os cars is shown
    timeTXT = Label(ui, text = "Detection time = -")
    timeTXT.grid(row=1, column=2)

    # The file menu in defined
    fileMenu = Menu(menuBar, tearoff = 0)
    fileMenu.add_command(label="Open", command = browseFiles)
    fileMenu.add_command(label="Save As", command = saveFile)
    menuBar.add_cascade(label = "File", menu = fileMenu)
    fileMenu.entryconfig("Save As", state = "disabled")

    # The car detection menu is defined
    carDetectionMenu = Menu(menuBar, tearoff = 0)
    carDetectionMenu.add_command(label="Yolo", command = lambda: CarDetectionOpt("Yolo"))
    carDetectionMenu.add_command(label = "Yolo Tiny", command =  lambda: CarDetectionOpt("Yolo Tiny"))
    # Yolo option is the default one
    menuBar.add_cascade(label = "Car detection", menu = carDetectionMenu)
    carDetectionMenu.entryconfig("Yolo", background = "gray82")

    # The LP detection menu is defined
    LPDetectionMenu = Menu(menuBar, tearoff=0)
    LPDetectionMenu.add_command(label = "Yolo", command = lambda: LPDetectionOpt("Yolo"))
    LPDetectionMenu.add_command(label = "Edge detection", command = lambda: LPDetectionOpt("Edge detection"))
    menuBar.add_cascade(label = "LP detection", menu = LPDetectionMenu)
    # Yolo option is the default one
    LPDetectionMenu.entryconfig("Yolo", background = "gray82")

    ui.config(menu = menuBar)
    ui.mainloop()