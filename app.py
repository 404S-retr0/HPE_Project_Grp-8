# Importing essential libraries
from flask import Flask, render_template, request

app = Flask(__name__)

#One-Hot_Encoding
import re
import numpy as np
import pandas as pd
global alphabet
global maxlen
global N_LANG
N_LANG=10
maxlen=13 #max word lenght 13
alphabet = "abcdefghijklmnopqrstuvwxyzíóéáñúüäßöàèêçôùîûâìòźåãõíłęążśćńøæ" #61 different character found in 10 European different Language
def one_hot_encode(data):
  char_to_int = dict((c, i) for i, c in enumerate(alphabet))
  integer_encoded = [char_to_int[char] for char in data]
  onehot_encoded = []
  for value in integer_encoded:
    letter = np.zeros(len(alphabet))
    letter[value] = 1
    onehot_encoded.append(letter)
  while(len(onehot_encoded)<maxlen):
    letter = np.zeros(len(alphabet))
    onehot_encoded.append(letter)
  return  np.array(onehot_encoded)

def encode_labels(label):
  temp = np.zeros(N_LANG)
  temp[label-1] = 1
  return temp

def process(test_str):
  test_str=test_str.lower()
  test_str = re.sub(r'[^a-zA-Z ]', '', test_str)
  return test_str

#Reading the dataset
x=[]
y=[]
df=pd.read_csv('https://github.com/404S-retr0/HPE_Project_Grp-8/blob/main/Data_Wordlists.csv?raw=true')
for word,lang in zip(df.WORDS,df.LANGAUAGE_VECTOR):
    if(pd.isna(word)):
      continue
    if(len(word)<=maxlen):
      x.append(one_hot_encode(process(word)))
      y.append(encode_labels(int(lang)))
x=np.array(x)
print(x.shape)
y=np.array(y)
print(y.shape)


#loading the model
from tensorflow import keras
from keras.models import model_from_json
json_file = open('modelupd.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("modelupd.h5")

#Defining text2list function
def text2list(text):
  import string
  import re
  test_str=text
  test_str = ''.join([i for i in test_str if not i.isdigit()]) 
  test_str=test_str.lower()
  punc = '''!()-[]{};:'"\,<>./¿?@#$%^&*_~+-=|`'''

  for ele in test_str:
    if ele in punc:
      test_str = test_str.replace(ele, "")  

  test_str_split=re.split('\s+', test_str)
  
  if (test_str_split[0]==''):
    test_str_split.remove('')
  if (test_str_split[len(test_str_split)-1]==''):
    test_str_split.remove('')

  return test_str_split

#Loading the Model and defining lang_detect in percentage Distribution
def lang_detect(text):
    list_of_word=text2list(text)
    k=[]
    for word in list_of_word:
        k.append(one_hot_encode(process(word)))
        ans=loaded_model.predict(np.array(k))
    print('Predication of language belonging of each unique word from a sentence\n')
    for q in range(len(list_of_word)):
        eng=0
        dan=0
        dut=0
        fre=0
        ger=0
        ita=0
        pol=0
        por=0
        spa=0
        swe=0
        eng+=ans[q][0]
        dan+=ans[q][1]
        dut+=ans[q][2]
        fre+=ans[q][3]
        ger+=ans[q][4]
        ita+=ans[q][5]
        pol+=ans[q][6]
        por+=ans[q][7]
        spa+=ans[q][8]
        swe+=ans[q][9]
        each_word=list_of_word[q]
        print(each_word)
        report = "Percentage Distribution\nEnglish:-{}\nDanish:-{}\nDutch:-{}\nFrench:-{}\nGerman:-{}\nItalian:-{}\nPolish:-{}\nPortuguese:-{}\nSpanish:-{}\nSwedish:-{}\n".format(eng,dan,dut,fre,ger,ita,pol,por,spa,swe)
        print(report)

@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method == 'GET':
	  return render_template('index.html')
  if request.method == 'POST':
    linetext = request.form.get("input") 
    return render_template('result.html', prediction= lang_detect(linetext))
   

if __name__ == '__main__':
	app.run()
