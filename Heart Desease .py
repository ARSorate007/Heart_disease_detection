import numpy as np
import matplotlib.pyplot as plt
from math import floor
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import the NeuralNetwork class from your neural_network module
from neural_network import NeuralNetwork

# Read the dataset
heart_df = pd.read_csv('dataset.csv')

# Data preprocessing and model training
class Train():
    def __init__(self):
        self.X = heart_df.drop(columns=['target'])

        # heart_df['target'] = heart_df['target'].replace(1, 0)
        # heart_df['target'] = heart_df['target'].replace(2, 1)

        self.y_label = heart_df['target'].values.reshape(self.X.shape[0], 1)

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y_label, test_size=0.2, random_state=2)

        # Standardizing the dataset
        self.sc = StandardScaler()
        self.sc.fit(self.Xtrain)
        self.Xtrain = self.sc.transform(self.Xtrain)
        self.Xtest = self.sc.transform(self.Xtest)

        self.nn = NeuralNetwork()  # Create the nn model
        self.nn.fit(self.Xtrain, self.ytrain)

        self.train_pred = self.nn.predict(self.Xtrain)
        self.test_pred = self.nn.predict(self.Xtest)

    # Set up text fields to display information
    def setTextField(self):
        shapeLabel = Label(window, text=f"Shape of train set is {self.Xtrain.shape}\nShape of test set is {self.Xtest.shape}\nShape of train label is {self.ytrain.shape}\nShape of test label is {self.ytest.shape}")
        trainLabel = Label(window, text="Train accuracy is {}".format(self.nn.acc(self.ytrain, self.train_pred)))  # Corrected labels
        testLabel = Label(window, text="Test accuracy is {}".format(self.nn.acc(self.ytest, self.test_pred)))  # Corrected labels
        shapeLabel.place(x=75, y=100)
        trainLabel.place(x=75, y=175)
        testLabel.place(x=75, y=200)

    def transformdata(self, data):
        return self.sc.transform(data)

    def predictOne(self, data):
        return self.nn.predict(data)

if __name__ == '__main__':
    def exit_function():
        exit()

    def predict():
        prediction = Tk()
        prediction.geometry("400x450")
        prediction.title("Heart Disease Prediction")

        Label(prediction, text= "Prediction", font=("arial", 16,"bold")).pack()
        Label(prediction, text= "Age:").place(x=50, y=35)
        Label(prediction, text="Sex:").place(x=50,  y= 60)
        Label(prediction, text="Chest Pain:").place(x=50, y=85)
        Label(prediction, text="Resting Blood Pressure:").place(x=50, y=110)
        Label(prediction, text="Serum Cholestrol:").place(x=50, y=135)
        Label(prediction, text="Fasting Blood Suger:").place(x=50, y=160)
        Label(prediction, text="Resting ECG result:").place(x=50, y= 185)
        Label(prediction, text="Max Hear Rate Achieved:").place(x=50, y=210)
        Label(prediction, text="Exercise Indused Agina:").place(x=50, y=235)
        Label(prediction, text="Old Peak:").place(x=50, y=260)
        Label(prediction, text="Slope of Peak :").place(x=50, y=285)
        Label(prediction, text="Number Of Major Vessels:").place(x=50, y=310)
        Label(prediction, text="Thal:").place(x=50, y=335)

        txAge = Entry(prediction)
        txSex = Entry(prediction)
        txChest =Entry(prediction)
        txResting = Entry(prediction)
        txSerum = Entry(prediction)
        txFasting = Entry(prediction)
        txECG = Entry(prediction)
        txHear =Entry(prediction)
        txExercise =Entry(prediction)
        txOldpeak =Entry(prediction)
        txSlope =Entry(prediction)
        txNumber =Entry(prediction)
        txThal =Entry(prediction)

        txAge.place(x=250, y=35)
        txSex.place(x=250, y=60)
        txChest.place(x=250, y=85)
        txResting.place(x=250, y=110)
        txSerum.place(x=250, y=135)
        txFasting.place(x=250, y=160)
        txECG.place(x=250, y=185)
        txHear.place(x=250, y=210)
        txExercise.place(x=250, y=235)
        txOldpeak.place(x=250, y=260)
        txSlope.place(x=250, y=285)
        txNumber.place(x=250, y=310)
        txThal.place(x=250, y=335)

        def getDataAndPredict():
            data=[[float(txAge.get()),float(txSex.get()),float(txChest.get()),float(txResting.get()),float(txSerum.get()),
                   float(txFasting.get()),float(txECG.get()),float(txHear.get()),float(txExercise.get()),float(txOldpeak.get()),
                   float(txSlope.get()),float(txNumber.get()),float(txThal.get())]]
            dataTransform = tr.transformdata(data)
            if floor(tr.predictOne(dataTransform)):
                Label(prediction, text="positive heart disease", font=("arial", 14, "bold")).place(x=150, y=400)
            else:
                Label(prediction, text="Negative heart disease", font=("arial", 14, "bold")).place(x=150, y=400)

        Button(prediction, text="Predict", command=getDataAndPredict).place(x= 75, y=400)

        prediction.mainloop()


    tr = Train()
    window = Tk()
    window.geometry("400x300")
    window.title("Heart Disease Prediction")

    label1 = Label(window, text="Heart Disease Prediction", font=("arial", 16, "bold")).pack()
    btnTraining = Button(window, text="Train Dataset", command=tr.setTextField)
    btnExit = Button(window, text="Exit", command=exit_function)
    btnTraining.place(x=75, y=50)
    btnInput = Button(window, text="Input Data", command=predict)  # Corrected button label and function
    btnInput.place(x=175, y=58)
    btnExit.place(x=350, y=250)

    window.mainloop()
