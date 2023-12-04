import fasttext
from iso_language_codes import *

##"tu madre wa puta"

def detectLanguage(option ,text):
    if(option == "1"):
        model = fasttext.load_model("lid.176.bin")

        predictions = model.predict(text, k=1)

        tmp = predictions[0][0]
        print(language_name(tmp[9:]))

    if(option == "2"):
        model = fasttext.train_supervised("C:/Users/bartek/Desktop/projektPJN/training_data.txt",
                                      epoch = 72,
                                      lr=0.5,
                                      dim = 200,
                                      ws = 5,
                                      loss="softmax")
        
        predictions = model.predict(text)
        language = predictions[0][0][9:]
        print(language)

    #predictions = model.predict(text, k=1)

    #tmp = predictions[0][0]
    #print(language_name(tmp[9:]))

option = input("\nWhat model do you want to use?\n1-predefined\n2-trained\n")
text = input("Put in a phrase to analyse: ")
detectLanguage(option, text)