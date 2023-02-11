#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Listas de librerias necesaria para crear un CHATBOT
# numpy manejo de datos (pandas)
# nltk Es una libreria que sirve para inerpretr lenguaje natural.
# tensorflow Libreria para Machine learning o aprendizaje automatico.
# tflearn sirve para realizar el deep learning 
# random Elegir algo al azar en este caso la respuesta a mostrar 


# In[1]:


pip install nltk


# In[2]:


pip install tensorflow


# In[3]:


pip install tflearn


# In[4]:


import nltk
nltk.download('punkt')


# In[5]:


import nltk 
from nltk.stem.lancaster import LancasterStemmer
import tensorflow
import tflearn
import pickle #sirve para guardar variables temporales de manera permanente
import random
import json
import numpy as np 

stremmer = LancasterStemmer('spanish') #Las 3 lineas anteriores se encarga de forma general adaptar el lenguaje 

with open("intents.json") as file:
    data = json.load(file)
    
#data impresion de prueba

try:
    with open("data.pickle","rb") as f:
        word, labels, training, output = pickle.load(f)
    
    with open('intents.json') as file:
        data2 = json.load(file)
    
    if data!=data2:
          raise Exception("El Archivo json ha cambiado") 
            
    print("Dentro del Try")
    print(word)
    print("")
    print("")
    print(labels)
    print("")
    print("")
    print(trainig)
    print("")
    print("")
    print(output)
    print("")
    print("")
    model.load("model.tflearn")
except:
    print("Dentro del Except")
    words=[] #Conjunto de palabras sin diferenciar a la frase que le pertenece... todo lo que contenga el tag 
    labels=[] #Titulos y leyendas   
    docs_x=[]
    docs_y=[]

    for intents in data['intents']:
        for patterns in intents['patterns']:
            wrds=nltk.word_tokenize(patterns)
            words.extend(wrds)  #extend lo que hace es pasa lo ultimo a una nueva lista
            #print(wrds)
            docs_x.append(wrds)
            docs_y.append(intents['tag'])

            if intents['tag'] not in labels:
                labels.append(intents['tag'])

    #wrds #impresion de prueba
    print (words)
    print ("")
    print ("")
    print ("")

    print (docs_x) #palarbras individuales y separadas por frases 

    print ("")
    print ("")
    print ("")

    print (docs_y) #palarbras individuales y separadas por tag

    print ("")
    print ("")
    print ("")

    print (labels) #palarbras individuales y separadas por tag


# In[6]:


print (words)
print ("")
print ("")
print ("")

words=[stremmer.stem(w.lower()) for w in words if w != "?"]

print (words)
print ("")
print ("")
print ("")

words = sorted(list(set(words)))
print (words)
print ("")


# In[ ]:


labels = sorted(labels)

print(labels)
#['greeting', 'goodbye', 'thanks', 'hours', 'payments', 'opentoday']

training=[]
output=[]

out_empty = [0 for _ in range (len(labels))]

#print(out_empty)
print ("")
print ("")

for x, doc in enumerate(docs_x):
    bag = []

    wrds=[stremmer.stem(w.lower())for w in doc]  #convertir la palabra a lenguaje natural

    for w in words:
        if w in wrds:
            #print("Entra por UNO")
            #print("Este es W")
            #print(w)
            #print("")
            #print("Este es wrds")
            #print(wrds)
            #print("")
            #print("Este es words")
            #print(words)
            #print("")
            bag.append(1)
        else:
            #print("Entra por DOS")
            #print("Este es W")
            #print(w)
            #print("")
            #print("Este es wrds")
            #print(wrds)
            #print("")
            #print("Este es words")
            #print(words)
            #print("")
            bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

    #Todo el codigo anterior es necesario para llegar a las dos 
    #variables "Finales" que alimentaran el sistema de machine
    #Learning llamadas training y output las cuales formaran 
    #parte de la capa de alimentacion. 

training = np.array(training)  #Contiene la informacion prepada con la cual se va a alimentar el sistema referentes a las palabras
output = np.array(output)      #Contiene la informacion preparada con la cual se va a alimentar el sistema referente a la categorizacion        

print(training) #palabras codificadas 0000
print("")
print("")
print(output)  #Categorizacion "tags" codificados 

with open ("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

#print(bag) indica si esta ese caracter en la lista


# In[ ]:


#tensorflow.reset_defaul_graph() #Es la primera vez que utilizo la libreria tensorflow en el codigoy estoy utilizando
#una funcion de esa libreria llamada reset_default_graph

tensorflow.compat.v1.reset_default_graph()

#Con esta linea estoy creando mi primera capa o capa 0 o capa de alimentacion 
net = tflearn.input_data(shape=[None, len(training[0])])

#Con esta linea estoy creando mi primera capa de red neuronal Circulos negros(imagen)
net = tflearn.fully_connected(net, 8)

#Con esta linea estoy creando mi primera capa de red neuronal Circulos rojos(imagen)
net = tflearn.fully_connected(net, 8)

#Continuacion
#Capa de decision circulos verdes 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

#Esta linea se encarga de construir el modelo final a partir de las especificaciones anteriores 
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    #Hasta  el momento hemos configurado nuestro modelo, es hhora de entrenarlo con nuestros datos
    #Par eso usaremos las siguientes lineas de codigo 
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# In[ ]:


def bag_of_words(s, words):
    #la funcion recibe dos parametros el primero es la frase que el usuario ingreso
    #El segundo es la bolsa de palabras creada previamente para alimentar el modelo
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s) #La funcion tokenize = convierte una frase a un conjunto de palabras
    s_words = [stremmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i]=1
                
    return np.array(bag)
    
def chat():
    print("Start talking with the boy (Type quit to stop)!") #mensaje inicial que muestral el chatbot
    while True: #se escucha constantemente al usuario 
        inp = input("You: ") #usuarios: ...mensaje
        if inp.lower() == "quit" or inp.lower() == "salir":
            break
            
        FraseCodificada = [bag_of_words(inp, words)]
        #El modelo me devuelve la probabilidad de que una frase ingresada por el usuario se encuentre en un tag -> categorizacion
        result = model.predict(FraseCodificada)
        results_index = np.argmax(result)
       
        print("Esta es la probabilidad mas alta:")
        print(results_index)
        
        tag = labels[results_index] #Se busca la categorizacion a la cual corresponde la frase ingresada
        
        for tg in data["intents"]:
            if tg['tag']==tag:
                responses = tg['responses']
                
        print(random.choice(responses))
        
        #print(result)
        #print(results_index)
chat()

