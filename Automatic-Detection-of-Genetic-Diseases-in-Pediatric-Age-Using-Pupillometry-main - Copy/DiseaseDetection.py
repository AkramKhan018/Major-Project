import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tkinter import messagebox
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import os
from sklearn.metrics import confusion_matrix
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout,Flatten
from sklearn.preprocessing import OneHotEncoder
import keras.layers
from sklearn.preprocessing import normalize

from keras.layers import Bidirectional
import tkinter
import json
import hashlib
from tkinter import messagebox
from tkinter import Label, Entry, Button
from tkinter import PhotoImage, Label



# Define the login function
def login():
    # Get the username and password from the entry widgets
    username = username_entry.get()
    password = password_entry.get()

    # Load registered users from file
    with open("registered_users.json", "r") as file:
        registered_users = json.load(file)

    # Check if the username and password are correct
    if username in registered_users and registered_users[username] == hash_password(password):
        # Close the login window
        main.destroy()  # Destroying the login window
        # Open main application or perform other actions upon successful login
    else:
        # Display an error message if the credentials are incorrect
        messagebox.showerror("Error", "Invalid username or password")

# Define the register function
def register():
    # Get the registration details from the entry widgets
    new_username = new_username_entry.get()
    new_password = new_password_entry.get()

    # Check if the username and password are provided
    if new_username == "" or new_password == "":
        messagebox.showerror("Error", "Please enter both username and password for registration.")
    else:
        # Check if the username already exists
        with open("registered_users.json", "r") as file:
            registered_users = json.load(file)
        if new_username in registered_users:
            messagebox.showerror("Error", "Username already exists. Please choose a different one.")
        else:
            # Add the new user to the registered users dictionary
            registered_users[new_username] = hash_password(new_password)
            save_registered_users(registered_users)
            messagebox.showinfo("Success", "Registration successful! You can now login with your new credentials.")

def hash_password(password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return hashed_password

def save_registered_users(registered_users):
    with open("registered_users.json", "w") as file:
        json.dump(registered_users, file)

def load_registered_users():
    try:
        with open("registered_users.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Toggle visibility of registration section
# Toggle visibility of registration section
def toggle_registration():
    if register_frame.winfo_ismapped():
        register_frame.place_forget()  # Hide the registration frame
        toggle_button.config(text="Not registered? Click here to register")
    else:
        register_frame.place(x=700, y=490, width=350, height=150)  # Show the registration frame
        toggle_button.config(text="Registered? Click here to hide")

def exit_application():
    main.destroy()
    sys.exit()



import tkinter
from tkinter import Entry, Label, Button, messagebox, Frame
import json
import hashlib

# Define the login function, register function, toggle_registration function, and other functions as before

# Create the login window
main = tkinter.Tk()
main.title("Login")

# Set window size and make it fullscreen
main.geometry("1300x700")
main.attributes('-fullscreen', True)

# Load background image
bg_image = tkinter.PhotoImage(file="maxresdefault.png")

# Create a Label widget to display the background image
background_label = tkinter.Label(main, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create and place the username label and entry widget
username_label = Label(main, text="Username:",  bg="lightblue",  bd=0, padx=5, pady=5, font=("Arial", 12))
username_label.place(x=700,y=150)
username_entry = Entry(main, font=("Arial", 16))
username_entry.place(x=800,y=150)

# Create and place the password label and entry widget
password_label = Label(main, text="Password:",  bg="lightblue",  bd=0, padx=5, pady=5, font=("Arial", 12))
password_label.place(x=700,y=200)
password_entry = Entry(main, show="*",font=("Arial", 16))
password_entry.place(x=800,y=200)

# Create and place the login button
login_button = Button(main, text="Login", command=login, width=15, bg="lightblue", bd=0, padx=5, pady=15, relief="ridge", font=("Arial", 12))
login_button.place(x=800,y=250)
# Create and place the registration button
toggle_button = Button(main, text="Not registered? Click here to register", command=toggle_registration, bg="lightgrey", bd=0, relief="ridge", font=("Arial", 12, "underline"))
toggle_button.place(x=750,y=450)
# Create registration frame
register_frame = tkinter.Frame(main)

# Create and place the registration label and entry widgets inside the registration frame
new_username_label = Label(register_frame, text="New Username:",  bg="lightblue",  bd=0, padx=5, pady=5, font=("Arial", 12))
new_username_label.grid(row=0, column=0, padx=15, pady=15)
new_username_entry = Entry(register_frame, font=("Arial", 15), width = 15)
new_username_entry.grid(row=0, column=1, padx=10, pady=5)

new_password_label = Label(register_frame, text="New Password:",  bg="lightblue",  bd=0, padx=5, pady=5, font=("Arial", 12))
new_password_label.grid(row=1, column=0, padx=15, pady=5)
new_password_entry = Entry(register_frame, show="*", font=("Arial", 15), width = 15)
new_password_entry.grid(row=1, column=1, padx=10, pady=5)

# Create and place the registration button inside the registration frame
register_button = Button(register_frame, text="Register", command=register, width=15, bg="lightgreen", bd=0, padx=5, pady=5, relief="ridge", font=("Arial", 12))
register_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5)


# Create and place the exit application button
exit_button = Button(main, text="Exit Application", command=exit_application, width=15, bg="lightcoral", bd=0, padx=5, pady=15, relief="ridge", font=("Arial", 12))
exit_button.place(x=620,y=683)

# Load registered users from file
registered_users = load_registered_users()



# Run the main event loop
main.mainloop()



main = tkinter.Tk()
main.title("Pupillary Patterns as Predictive Keys to detect Inherited Diseases in Infants")
main.geometry("1300x1200")

global filename

global classifier
global left_X_train, left_X_test, left_y_train, left_y_test
global right_X_train, right_X_test, right_y_train, right_y_test

global left_X, left_Y

global left_pupil
global right_pupil
global count
global left
global right
global ids
global left_svm_acc
global right_svm_acc
global left_classifier
global right_classifier
global classifier
global ensemble_acc
global elm_acc
global lstm_acc,bilstm_acc



def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Pupillometric  dataset loaded\n')

def filtering():
    global left_pupil
    global right_pupil
    global count
    global left
    global right
    global ids
    left_pupil = []
    right_pupil = []
    count = 0
    left = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    right = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    ids = 1
    for root, dirs, directory in os.walk('dataset'):
        for i in range(len(directory)):
            filedata = open('dataset/'+directory[i], 'r')
            lines = filedata.readlines()
            left_pupil.clear()
            right_pupil.clear()
            count = 0
            for line in lines:
                line = line.strip()
                arr = line.split("\t")
                if len(arr) == 8:
                    if arr[7] == '.....':
                        left_pupil.append(float(arr[3].strip()))
                        right_pupil.append(float(arr[6].strip()))
                        count = count + 1;
                        if count == 100:
                            left_minimum = min(left_pupil)
                            right_minimum = min(right_pupil)
                            left_maximum = max(left_pupil)
                            right_maximum = max(right_pupil)
                            left_delta =  left_maximum - left_minimum
                            right_delta = right_maximum - right_minimum
                            left_CH = left_delta / left_maximum
                            right_CH = right_delta / right_maximum
                            latency = 0.5
                            left_MCV = left_delta/(left_minimum - latency)
                            right_MCV = right_delta/(right_minimum - latency)
                            count = 0
                            left_pupil.clear()
                            right_pupil.clear()
                            if left_minimum > 500 and left_maximum > 500:
                                left+=str(ids)+","+str(left_maximum)+","+str(left_minimum)+","+str(left_delta)+","+str(left_CH)+","+str(latency)+","+str(left_MCV)+",1\n"
                            else:
                                left+=str(ids)+","+str(left_maximum)+","+str(left_minimum)+","+str(left_delta)+","+str(left_CH)+","+str(latency)+","+str(left_MCV)+",0\n"
                            if right_minimum > 500 and right_maximum > 500:
                                right+=str(ids)+","+str(right_maximum)+","+str(right_minimum)+","+str(right_delta)+","+str(right_CH)+","+str(latency)+","+str(right_MCV)+",1\n"
                            else:
                                right+=str(ids)+","+str(right_maximum)+","+str(right_minimum)+","+str(right_delta)+","+str(right_CH)+","+str(latency)+","+str(right_MCV)+",0\n"
                            ids = ids + 1
            filedata.close()
    
    text.delete('1.0', END)
    text.insert(END,'Features filteration process completed\n')
    text.insert(END,'Total patients found in dataset : '+str(ids)+"\n")
    
def featuresExtraction():
    f = open("left.txt", "w")
    f.write(left)
    f.close()
    f = open("right.txt", "w")
    f.write(right)
    f.close()
    text.delete('1.0', END)
    text.insert(END,'Both eye pupils extracted features saved inside left.txt and right.txt files \n')
    text.insert(END,"Extracted features are \nPatient ID, MAX, MIN, Delta, CH, Latency, MDV, CV and MCV\n")

def featuresReduction():
    text.delete('1.0', END)
    global left_X, left_Y
    global left_X_train, left_X_test, left_y_train, left_y_test
    global right_X_train, right_X_test, right_y_train, right_y_test
    left_pupil =  pd.read_csv('left.txt')
    right_pupil =  pd.read_csv('right.txt')
    cols = left_pupil.shape[1]

    left_X = left_pupil.values[:, 1:(cols-1)] 
    left_Y = left_pupil.values[:, (cols-1)]

    right_X = right_pupil.values[:, 1:(cols-1)] 
    right_Y = right_pupil.values[:, (cols-1)]

    indices = np.arange(left_X.shape[0])
    np.random.shuffle(indices)
    left_X = left_X[indices]
    left_Y = left_Y[indices]

    indices = np.arange(right_X.shape[0])
    np.random.shuffle(indices)
    right_X = right_X[indices]
    right_Y = right_Y[indices]

    left_X = normalize(left_X)
    right_X = normalize(right_X)
    

    left_X_train, left_X_test, left_y_train, left_y_test = train_test_split(left_X, left_Y, test_size = 0.2,random_state=42)
    right_X_train, right_X_test, right_y_train, right_y_test = train_test_split(right_X, right_Y, test_size = 0.2,random_state=42)

    text.insert(END,"Left pupil features training size : "+str(len(left_X_train))+" & testing size : "+str(len(left_X_test))+"\n")
    text.insert(END,"Right pupil features training size : "+str(len(right_X_train))+" & testing size : "+str(len(right_X_test))+"\n")

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Diameter')
    plt.plot(left_pupil['MAX'], 'ro-', color = 'indigo')
    plt.plot(right_pupil['MAX'], 'ro-', color = 'green')
    plt.legend(['Left Pupil', 'Right Pupil'], loc='upper left')
    plt.title('Pupil Diameter Graph')
    plt.show()
    

    
    
    

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	

    
    
def rightSVM():
    global right_classifier
    text.delete('1.0', END)
    global right_svm_acc
    temp = []
    for i in range(len(right_y_test)):
        temp.append(right_y_test[i])
    temp = np.asarray(temp)    
    right_classifier = svm.SVC()
    right_classifier.fit(right_X_train, right_y_train)
    text.insert(END,"Right pupil SVM Prediction Results\n") 
    prediction_data = prediction(right_X_test, right_classifier) 
    right_svm_acc = accuracy_score(temp,prediction_data)*100
    text.insert(END,"Right pupil SVM Accuracy : "+str(right_svm_acc)+"\n")

    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil SVM Algorithm Specificity : '+str(specificity)+"\n")

def leftSVM():
    global left_classifier
    text.delete('1.0', END)
    global left_svm_acc
    temp = []
    for i in range(len(left_y_test)):
        temp.append(left_y_test[i])
    temp = np.asarray(temp) 
    left_classifier = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)
    left_classifier.fit(left_X_train, left_y_train)
    text.insert(END,"Left pupil SVM Prediction Results\n") 
    prediction_data = prediction(left_X_test, left_classifier) 
    left_svm_acc = accuracy_score(temp,prediction_data)*100
    text.insert(END,"Left pupil SVM Accuracy : "+str(left_svm_acc)+"\n")

    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Left pupil SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Left pupil SVM Algorithm Specificity : '+str(specificity)+"\n")

def ensemble():
    global classifier
    global ensemble_acc
    text.delete('1.0', END)

    trainX = np.concatenate((right_X_train, left_X_train))
    trainY = np.concatenate((right_y_train, left_y_train))

    testX = np.concatenate((right_X_test, left_X_test))
    testY = np.concatenate((right_y_test, left_y_test))

    left_classifier = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
    right_classifier = svm.SVC(kernel='linear', class_weight='balanced', probability=True)

    temp = []
    for i in range(len(testY)):
        temp.append(testY[i])
    temp = np.asarray(temp) 

    classifier = VotingClassifier(estimators=[
         ('SVMLeft', left_classifier), ('SVMRight', right_classifier)], voting='hard')
    classifier.fit(trainX, trainY)
    text.insert(END,"Optimized Ensemble Prediction Results\n") 
    prediction_data = prediction(testX, classifier) 
    ensemble_acc =  (accuracy_score(temp,prediction_data)*100)
    text.insert(END,"Ensemble OR Accuracy : "+str(ensemble_acc)+"\n")

    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil Ensemble OR SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil Ensemble OR SVM Algorithm Specificity : '+str(specificity)+"\n")


def runLSTM():
    global lstm_acc
    global left_X, left_Y

    Y = left_Y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y)
    X = left_X.reshape((left_X.shape[0], left_X.shape[1], 1))
    print(Y)
    print(X.shape)
    
    model = Sequential()
    model.add(keras.layers.LSTM(32,input_shape=(X.shape[1], 1)))#defining LSTM with input dataset size and number of filters as 32 for first layer
    model.add(Dropout(0.5)) #while filtering dataset Dropout will remove all unrelated or irrelevant dataset and hold only important features from dataset
    model.add(Dense(32, activation='relu')) #creating another layer with 32 filetrs
    model.add(Dense(2, activation='softmax')) #creating output prediction layer with number of output as 3 (HIGH, LOW or MEDIUM)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#compiling the model and asking to calculate accuracy for each iteration
    hist = model.fit(X, Y, verbose=2, batch_size=5, epochs=10)#start training model with batch size 5 and epoch as 100 with X and Y input data
    accuracy = hist.history
    acc = accuracy['accuracy']                
    lstm_acc = acc[9] * 100
    text.insert(END,"\nLSTM Accuracy : "+str(lstm_acc)+"\n\n")
    text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')
    print(model.summary())
    prediction_data = model.predict(X)
    temp = Y.argmax(axis=1)
    prediction_data = prediction_data.argmax(axis=1)
    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil LSTM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil LSTM Algorithm Specificity : '+str(specificity)+"\n")
    

def runBILSTM():
    global bilstm_acc
    global left_X, left_Y

    Y = left_Y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y)
    X = left_X.reshape((left_X.shape[0], left_X.shape[1], 1))
    print(Y)
    print(X.shape)
    model = Sequential()
    model.add(Bidirectional(keras.layers.LSTM(64, return_sequences=True,input_shape=(X.shape[1], 1))))
    model.add(Bidirectional(keras.layers.LSTM(32, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#compiling the model and asking to calculate accuracy for each iteration
    hist = model.fit(X, Y, verbose=2, batch_size=5, epochs=10)#start training model with batch size 5 and epoch as 100 with X and Y input data
    accuracy = hist.history
    acc = accuracy['accuracy']                
    bilstm_acc = acc[9] * 100
    text.insert(END,"\nBI-LSTM Accuracy : "+str(bilstm_acc)+"\n\n")
    text.insert(END,'Bi-LSTM Model Summary can be seen in black console for layer details\n')
    print(model.summary())

    prediction_data = model.predict(X)
    temp = Y.argmax(axis=1)
    prediction_data = prediction_data.argmax(axis=1)
    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil BI-LSTM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil BI-LSTM Algorithm Specificity : '+str(specificity)+"\n")

def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "testData")
    test = pd.read_csv(filename)
    test = test.values[:, 0:7]
    total = len(test)
    text.insert(END,filename+" test file loaded\n");
    y_pred = classifier.predict(test)
    for i in range(len(test)):
        print(str(y_pred[i]))
        if str(y_pred[i]) == '0.0':
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No disease detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Disease detected')+"\n\n")


def extension():
    global elm_acc
    text.delete('1.0', END)

    trainX = np.concatenate((right_X_train, left_X_train))
    trainY = np.concatenate((right_y_train, left_y_train))

    testX = np.concatenate((right_X_test, left_X_test))
    testY = np.concatenate((right_y_test, left_y_test))

    srhl_tanh = MLPRandomLayer(n_hidden=100, activation_func='tanh')
    classifier = GenELMClassifier(hidden_layer=srhl_tanh)
    classifier.fit(trainX, trainY)

    
    text.insert(END,"Extension Extreme Learning Machine Prediction Results\n") 
    prediction_data = prediction(testX, classifier)
    #for i in range(0,(len(testY)-30)):
    #    prediction_data[i] = testY[i]
    elm_acc =  (accuracy_score(testY,prediction_data)*100)
    text.insert(END,"Extension Extreme Learning Machine Accuracy : "+str(elm_acc)+"\n")

    cm = confusion_matrix(testY, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil ELM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil ELM Algorithm Specificity : '+str(specificity)+"\n")
    

def graph():
    height = [right_svm_acc,left_svm_acc,ensemble_acc,elm_acc,lstm_acc,bilstm_acc]
    bars = ('Right Pupil SVM Acc','Left Pupil SVM Acc','Ensemble OR (L & R Pupil) Acc','ELM Acc','LSTM Acc','BI-LSTM Acc')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
font = ('times', 16, 'bold')
title = Label(main, text='Pupillary Patterns as Predictive Keys to detect Inherited Diseases in Infants')
title.config(bg='goldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Pupillometric Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='medium sea green', fg='Black')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

filterButton = Button(main, text="Run Filtering", command=filtering)
filterButton.place(x=700,y=200)
filterButton.config(font=font1) 

extractButton = Button(main, text="Run Features Extraction", command=featuresExtraction)
extractButton.place(x=700,y=250)
extractButton.config(font=font1) 

featuresButton = Button(main, text="Run Features Reduction", command=featuresReduction)
featuresButton.place(x=700,y=300)
featuresButton.config(font=font1)

rightsvmButton = Button(main, text="Run SVM on Right Eye Features", command=rightSVM)
rightsvmButton.place(x=700,y=350)
rightsvmButton.config(font=font1)

leftsvmButton = Button(main, text="Run SVM on Left Eye Features", command=leftSVM)
leftsvmButton.place(x=700,y=400)
leftsvmButton.config(font=font1)


ensembleButton = Button(main, text="Run OR Ensemble Algorithm (Left & Right SVM)", command=ensemble)
ensembleButton.place(x=700,y=450)
ensembleButton.config(font=font1)

extensionButton = Button(main, text="Run Extension Extreme Learning Machine Algorithm", command=extension)
extensionButton.place(x=700,y=500)
extensionButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM", command=runLSTM)
lstmButton.place(x=700,y=550)
lstmButton.config(font=font1)

bilstmButton = Button(main, text="Run BILSTM", command=runBILSTM)
bilstmButton.place(x=850,y=550)
bilstmButton.config(font=font1)


graphButton = Button(main, text="Accuracy Graph with Metrics", command=graph)
graphButton.place(x=700,y=600)
graphButton.config(font=font1)


predictButton = Button(main, text="Predict Disease", command=predict)
predictButton.place(x=700,y=650)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)



main.config(bg='SteelBlue4')

main.mainloop()
