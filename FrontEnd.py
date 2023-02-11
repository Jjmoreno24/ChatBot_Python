import tkinter
import re
import pandas as pd
import nltk
import tensorflow
import tflearn
import random
import numpy as np
import pickle
from tkinter import *
from PIL import Image, ImageTk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
import json
from tensorflow.python.framework import ops


stemmer=SnowballStemmer('spanish')

#def run_chatbot():



def ventana():
    print("ENTRA AL CHATBOT")

    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)  # Convierto la frase ingresada por el usuario en palabras
        s_words = [stemmer.stem(word.lower()) for word in s_words]
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)

    def chatbot_response(msg):
        import json
        with open('intents.json') as file:
            data = json.load(file)

        try:
            with open("data.pickle", "rb") as f:
                words, labels, training, output, data = pickle.load(f)

            with open('intents.json') as file:
                data2 = json.load(file)

            if data != data2:
                raise Exception("El Archivo json ha cambiado")
            model.load("model.tflearn")

        except:
            print("Estoy dentro del EXCEPT")

            words = []  # Palabras sin deferenciar la frase a la que pertenecen
            labels = []  # Titulos, legendas.
            docs_x = []
            docs_y = []

            for intents in data['intents']:
                for patterns in intents['patterns']:
                    wrds = nltk.word_tokenize(patterns)  # Convierte una frase a un conjunto de palabras
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intents["tag"])

                    if intents['tag'] not in labels:
                        labels.append(intents['tag'])

            words = [stemmer.stem(w.lower()) for w in words if w != "?"]

            words = sorted(list(set(words)))  # Organizando el conjunto de paralabras de forma no repetiva y ordenada.

            labels = sorted(labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(labels))]

            # Este ciclo for se encarga de analizar todas y cada una de las palabras en todas y cada una de las frases

            for x, doc in enumerate(docs_x):
                bag = []

                wrds = [stemmer.stem(w.lower()) for w in doc]

                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                    output_row = out_empty[:]
                    output_row[labels.index(docs_y[x])] = 1

                    training.append(bag)
                    output.append(output_row)

            training = np.array(
                training)  # Contiene la informacion preparada con la cual se va a alimentar el sistema referentes a las palabras
            output = np.array(
                output)  # Contiene la informacion preparada con la cual se va a alimentar el sistema referente a la categorizacion

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output, data), f)

        tensorflow.compat.v1.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)

        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        try:
            model.load("model.tflearn")
        except:
            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            model.save("model.tflearn")

        results = model.predict([bag_of_words(msg, words)])
        results_index = np.argmax(results)  # La funcion argmax obtiene la probabilidad mas alta.

        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return (random.choice(responses))

    def send():
        msg = EntryBox.get("1.0", 'end-1c').strip()
        print("mensaje captado: ", EntryBox.get("1.0", "end-1c"))

        print("\nborrando....")
        EntryBox.delete("0.0", END)
        print("mensaje borrado: ", EntryBox.get("1.0", "end-1c"))

        res = chatbot_response(msg)

        if msg != "":
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + " \n\n")
            ChatLog.config(foreground="black", font=("Verdana", 12))
            ChatLog.insert(END, "ChatBot: " + res + " \n\n")
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

    base = Tk()
    base.title("Chatbot (Atención al usuario)")
    base.geometry("400x500")
    base.config(bg="#FAA4A4")
    base.resizable(width=FALSE, height=FALSE)  # Mantiene fija la ventana.

    ChatLog = Text(base, bd=0, bg="white", width=8, height="50", font="Arial")
    ChatLog.config(foreground="Black", font=("Verdana", 12) )  # foreground cambia el color de la letra
    ChatLog.insert(END, "Bienvenido\n\n")
    ChatLog.place(x=6, y=6, height=386, width=370)
    ChatLog.config(state=DISABLED)  # bloquea la entrada de texto(lo hace de solo lectura)

    scrollbar = Scrollbar(base, command=ChatLog.yview(), cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set
    scrollbar.place(x=376, y=6, height=386)

    EntryBox = Text(base, bg="white", height="5", width=29, font="Arial")
    EntryBox.place(x=6, y=401, height=90, width=265)



    SendButton = Button(base, font=("Verdana", 12, "bold"), text="Send", height="5", width="9",
                    bd=0, bg="black", activebackground="dark red", fg='#FAA4A4', command=send)
    SendButton.place(x=282, y=401, height=90)

    base.bind('<Return>', lambda event: send())

    #  Primer parameter: posición
    #  ChatLog.grid(column=0, row=0)

    base.mainloop()

names = []
emails = []
phones = []
subs = []
informe = pd.DataFrame()
headers = ["Name", "eMail", "Number", "Subscribed"]


# _______Validaciones______
def es_nombre(nombre):
    nombre = nombre.replace(' ', '')

    if nombre.isalpha() == False:
        nombre = ""
        raise ValueError()


# ___ Telefono ___
def validar_telefono(numero):
    patron = re.compile(r'^\d{4}\d{4}$')

    return patron.match(numero)


# ____ Correo ____
def validar_correo(correo):
    print("está validando")
    patron = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if re.match(patron, correo):
        return True
        print("validó el correo")
    else:
        return False

# ______ Guardar el archivo en csv _________

def save_to_csv(email, name, tel, subscribed):
    dict = {'Name': '', 'eMail': '', 'Phone': 0, 'Subscribed': ''}

    dict['Name'] = name
    dict['eMail'] = email
    dict['Phone'] = tel
    dict['Subscribed'] = subscribed

    try:
        df = pd.DataFrame(dict, index=[0])
        df.to_csv("informe.csv", sep=";", mode='a', index=False, header=False)
        pass

    except:
        df = pd.DataFrame(dict, index=[0])
        df.to_csv("informe.csv", sep=";", index=False)
        pass

# _________ validar correo y telefono ________
def validate():
    email = correo_in.get("1.0", "end-1c").strip()
    name = nombre_in.get("1.0", "end-1c").strip()
    tel = tel_in.get("1.0", "end-1c").strip()

    if checkvar.get() == 0:
        subscribed = 'No'
    else:
        subscribed = 'yes'

    if validar_correo(email):
        print("CORREO VALIDO")
        validate_mail = True
    else:
        tkinter.messagebox.showerror(message="Correo inválido", title="ERROR")
        print("CORREO INVALIDO")
        validate_mail = False
    if validar_telefono(tel):
        print("NUMERO VALIDO")
        validate_phone = True
    else:
        print("NUMERO INVÁLIDO")
        tkinter.messagebox.showerror(
            message="Numero de teléfono inválido, no incluyas letras ni caracteres especiales",
            title="ERROR")
        validate_phone = False

    if (validate_phone and validate_mail):
        print("datos guardados...")
        save_to_csv(email, name, tel, subscribed)  #Guarda
        log_in.destroy()                           #Cierra la ventana del LOGIN
        ventana()                                #Corre el chatbot
        print("\n○\nLanzando chatbot...")



# ________________________________________________ front end _________________________________________
log_in = Tk()
log_in.geometry("550x500")
log_in.title("Inicia sesión")
log_in.resizable(width=TRUE, height=TRUE)

# ______ Imagen Background _______
image = PhotoImage(file="background.png")
background_label = Label(log_in, image=image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
texto1 = "introduce tu nombre"

# ______ Label que indica al usuario que ingrese el nombre _______

nombre_lab = Label(log_in, compound=CENTER, text=texto1, image=image, padx=100, pady=70)
nombre_lab.config(foreground="black", font=("Montserrat, sans-serif", "12", "bold"))
nombre_lab.place(x=100, y=70, height=25, width=170)

# ______ Usuario ingresa el nombre _______
nombre_in = Text(log_in, bd=0, bg="white", width=1, height="5", font="Arial")
nombre_in.config(foreground="Black", font=("Verdana", 12))
nombre_in.place(x=100, y=100, height=25, width=350)

# ______ Label que indica al usuario que ingrese el correo _______

correo_lab = Label(log_in, width=1, height="5", font="Arial", compound=CENTER, image=image,
     text="Introduce tu correo electronico")
correo_lab.config(foreground="black", font=("Montserrat, sans-serif", 12, "bold"), bd=1,  background="ghost white")
correo_lab.place(x=100, y=145, height=25, width=240)

w_mail = Label(log_in, foreground="black", width=1, height="5", font="Arial", background="ghost white", compound=CENTER, image=image,
               text="(nombre@dominio.com)")
w_mail.config(foreground="Black", font=("Montserrat, sans-serif", 7))
w_mail.place(x=350, y=150, height=20, width=100)

# ______ Usuario ingresa el correo _______
correo_in = Text(log_in, bd=0, bg="white", width=1, height="5", font="Arial")
correo_in.config(foreground="Black", font=("Verdana", 12))
correo_in.place(x=100, y=175, height=25, width=350)

# ______ Label que indica al usuario que ingrese el numero de telefono _______

tel_lab = Label(log_in, bd=0, bg="ghost white", width=1, height="5", font="Arial", compound=CENTER, image=image,
                text="Introduce tu numero de telefono")
tel_lab.config(foreground="Black", font=("Montserrat, sans-serif", 12, "bold"))
tel_lab.place(x=100, y=220, height=25, width=253)

w_tel = Label(log_in, bd=0, bg="ghost white", width=1, height="5", font="Arial", compound=CENTER, image=image,
              text="(xxxxxxxx)")
w_tel.config(foreground="Black", font=("Montserrat, sans-serif", 8))
w_tel.place(x=370, y=220, height=25, width=60)

# ______ Usuario ingresa el numero de telefono _______
tel_in = Text(log_in, bd=0, bg="white", width=1, height="5", font="Arial")
tel_in.config(foreground="Black", font=("Verdana", 12))
tel_in.place(x=100, y=250, height=25, width=350)

# __________ SMI Logo ___________

smi_logo = ImageTk.PhotoImage(Image.open("LogoC.png"))
smi_label = Label(log_in, image=smi_logo, background="#FAA4A4", compound=CENTER)
smi_label.place(x=100, y=350)

# _______ Checkvar ___________

checkvar = IntVar()
check_btn = tkinter.Checkbutton(log_in, text="¿Deseas estar suscribirte a nuestro portal?",
                                variable=checkvar)
check_btn.config(background="#fa8072", font=("Montserrat, sans-serif", 9))
check_btn.place(x=95, y=300)

# _______ Botón de enviar ___________
sendButton = Button(log_in, font=("Montserrat, sans-serif", 12, "bold"),
                    text="Enviar", height="5", width="9",
                    bd=0, bg="black", activebackground="dark red",
                    fg='#FAA4A4', command=validate)
sendButton.place(x=350, y=350, height=90)

# ______ Crear dataframe para guardar los datos ________


log_in.mainloop()