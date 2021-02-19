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
df=pd.read_csv('https://github.com/404S-retr0/HPE_Project_Grp-8/blob/web_design-via-Flask-API/Data_Wordlists.csv?raw=true')
for word,lang in zip(df.WORDS,df.LANGAUAGE_VECTOR):
    if(pd.isna(word)):
      continue
    if(len(word)<=maxlen):
      x.append(one_hot_encode(process(word)))
      y.append(encode_labels(int(lang)))
x=np.array(x)
y=np.array(y)

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

  test_str_split=re.split("\s+", test_str)
  if (test_str_split[0]==''):
    test_str_split.remove('')
  if (test_str_split[len(test_str_split)-1]==''):
    test_str_split.remove('')

  return test_str_split
  
#Loading the Model and defining lang_detect in percentage Distribution
#loading the model
from keras.models import model_from_json
json_file = open('modelupd.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("modelupd.h5")

def lang_detect(text):
    list_of_word=text2list(text)
    k=[]
    
    for word in list_of_word:
        k.append(one_hot_encode(process(word)))
        ans=loaded_model.predict(np.array(k))
    print('Predication of language belonging of each unique word from a sentence\n')
    result=""
    result+='''<center><font face = "Comic sans MS" color="blue"><h1>Predication of language belonging of each unique word from a sentence</h1></font></center>
              <center><font color="green"><h2>Text = {}</h2><br></font></center>
            '''.format(text)
    teng=0
    tdan=0
    tdut=0
    tfre=0
    tger=0
    tita=0
    tpol=0
    tpor=0
    tspa=0
    tswe=0
    for q in range(len(list_of_word)):
        eng=ans[q][0]
        dan=ans[q][1]
        dut=ans[q][2]
        fre=ans[q][3]
        ger=ans[q][4]
        ita=ans[q][5]
        pol=ans[q][6]
        por=ans[q][7]
        spa=ans[q][8]
        swe=ans[q][9]
        each_word=list_of_word[q]
        teng+=eng
        tdan+=dan
        tdut+=dut
        tfre+=fre
        tger+=ger
        tita+=ita
        tpol+=pol
        tpor+=por
        tspa+=spa
        tswe+=swe
        #in cmd
        print(each_word)
        report = "Percentage Distribution\nEnglish:-{}\nDanish:-{}\nDutch:-{}\nFrench:-{}\nGerman:-{}\nItalian:-{}\nPolish:-{}\nPortuguese:-{}\nSpanish:-{}\nSwedish:-{}\n".format(eng,dan,dut,fre,ger,ita,pol,por,spa,swe)
        print(report)
        #in html template(result.html)
        result += '''
              <center><font color="#CC3433"><h2>word = {}</h2></font></center><br>
              <font face = "Times New Roman" ><h4>Percentage Distribution</h4></font>
              English:-{}<br>
              Danish:-{}<br>
              Dutch:-{}<br>
              French:-{}<br>
              German:-{}<br>
              Italian:-{}<br>
              Polish:-{}<br>
              Portuguese:-{}<br>
              Spanish:-{}<br>
              Swedish:-{}<br><br>
              '''.format(each_word,round(eng*100,2),round(dan*100,2),round(dut*100,2),round(fre*100,2),round(ger*100,2),round(ita*100,2),round(pol*100,2),round(por*100,2),round(spa*100,2),round(swe*100,2))
    
    teng/=len(list_of_word)
    tdan/=len(list_of_word)
    tdut/=len(list_of_word)
    tfre/=len(list_of_word)
    tger/=len(list_of_word)
    tita/=len(list_of_word)
    tpol/=len(list_of_word)
    tpor/=len(list_of_word)
    tspa/=len(list_of_word)
    tswe/=len(list_of_word)
    teng=round(teng*100,2)
    tdan=round(tdan*100,2)
    tdut=round(tdut*100,2)
    tfre=round(tfre*100,2)
    tger=round(tger*100,2)
    tita=round(tita*100,2)
    tpol=round(tpol*100,2)
    tpor=round(tpor*100,2)
    tspa=round(tspa*100,2)
    tswe=round(tswe*100,2)
    result += '''
              <br><br><center><font color="#ff6666"><h2>Total Percentage Distribution</h2></font></center><br>
              Total English:-{}<br>
              Total Danish:-{}<br>
              Total Dutch:-{}<br>
              Total French:-{}<br>
              Total German:-{}<br>
              Total Italian:-{}<br>
              Total Polish:-{}<br>
              Total Portuguese:-{}<br>
              Total Spanish:-{}<br>
              Total Swedish:-{}<br><br>
              '''.format(teng,tdan,tdut,tfre,tger,tita,tpol,tpor,tspa,tswe)
    return result

@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method == 'GET':
	  return render_template('input.html')
  
  linetext = request.form.get("input") 
  return lang_detect(linetext)
   

if __name__ == '__main__':
	app.run()
