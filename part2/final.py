import numpy as np
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emoji import UNICODE_EMOJI
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
from lime.lime_text import LimeTextExplainer
import lime
import lime.lime_text
import seaborn as sb
from matplotlib import transforms
import imgkit
from fpdf import FPDF
import webbrowser
import sys

analyser = SentimentIntensityAnalyzer()
  
def is_emoji(s):
    return s in UNICODE_EMOJI

filename = 'model.sav'
clf=joblib.load(filename)

def predict_prob(x):
    out=[]
    for i in x:
        i=i.split()
        total=np.array([0,0,0]);den=0
        for j in i:
            if is_emoji(j):
                j=emoji.demojize(j)[1:-1]
                j = j.replace('_',' ')
            s=analyser.polarity_scores(j)
            if not (s['neg']==0 and s['pos']==0 and s['neu']==1):
                total[0]+=s['pos'];total[1]+=s['neg'];total[2]+=s['neu']
                den+=1
        if den!=0:
            total=total/den
        p=clf.predict_proba([[total[0], total[1], total[2]]])
        if ('#irony' in i) or ('#not' in i) or ('#sarcasm' in i) or ('#Irony' in i) or ('#Not' in i) or ('#Sarcasm' in i) or ('irony' in i) or ('sarcasm' in i) :
            out.append(np.array([[p[0][0]/2, p[0][1]+p[0][0]/2]]))
        else:
            out.append(p)
    return np.asarray(out).reshape([len(x), 2])

def my_color_func(word, *args, **kwargs):
    s=analyser.polarity_scores(word)
    if s['pos']!=0 and s['neg']==0:
        color = '#00ff00' # green
    elif s['pos']==0 and s['neg']!=0:
        color = '#ff0000' # red
    else:
        color = '#0000ff' # blue
    return color

def word_cloud(text):
    tuples={};den=0
    for i in text.split():
        s=analyser.polarity_scores(i)
        if s['pos']!=0 :
            tuples[i]= s['pos']
        elif s['neg'] !=0:
            tuples[i]= s['neg']
        else:
            tuples[i]=0
        den+= tuples[i]  
    if den==0:
        tuples['No Irony']=1;den=1
    for i in text.split():
        if tuples[i]!=0:
            tuples[i]/=den
        else:
            del tuples[i]
    
    wordcloud = WordCloud(color_func=my_color_func)
    wordcloud = WordCloud().generate_from_frequencies(tuples)
    fig=plt.figure()
    plt.imshow(wordcloud.recolor(color_func=my_color_func), interpolation="bilinear")
    plt.title('Word cloud showing positive(green) and negative words(red)')
    plt.axis("off")
    plt.gcf()
    plt.savefig('fig4.jpg',format='jpg')
tr = transforms.Affine2D().rotate_deg(90)

def heat_map(text):
    text=text.split()
    data=[]
    for i in text:
        s=analyser.polarity_scores(i)
        data.append(s['pos']-s['neg'])
    labels=np.array(text).reshape((1,len(text)))
    data=np.array(data).reshape(1,(len(text)))
    fig, ax = plt.subplots(figsize=(67,2))
    ax= sb.heatmap(data, annot=labels, fmt='', linewidths=.5, cmap='RdBu', annot_kws={"size": 25})
    plt.imshow(tr)
    plt.axis("off")
    plt.gcf()
    plt.savefig('fig5.jpg',format='jpg')    
    
def main(text):
    temp=predict_prob([text])
    if temp[0][1]>=0.5:
        fig0='irony.jpg'
    else:
        fig0='non-ironic.jpg'
    explainer = LimeTextExplainer(class_names=['Non-Ironic', 'Ironic'])
    exp = explainer.explain_instance(text, predict_prob, num_features=4)
    exp.as_pyplot_figure()
    plt.gcf()
    plt.savefig('fig1.jpg',format='jpg')
    x=exp.as_list()
    dic={}
    for i in range(len(x)):
        dic[x[i][0]]=abs(x[i][1])
    wordcloud = WordCloud()
    wordcloud = WordCloud().generate_from_frequencies(dic)
    fig=plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title('Word cloud of the most important words')
    plt.gcf()
    plt.savefig('fig2.jpg',format='jpg')
    exp.save_to_file('foo.html')
    config = imgkit.config(wkhtmltoimage=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe")
    imgkit.from_file('foo.html', 'fig3.jpg', config=config)
    word_cloud(text)
    heat_map(text)
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=13)
    pdf.cell(270, 2, txt="Prediction:", ln=1, align="C")
    pdf.image(fig0,x=135,y=15,w=45)
    pdf.cell(30, 90, txt="Label Analysis:", ln=1, align="C")
    pdf.image('fig1.jpg',x=20,y=62,w=90)
    pdf.image('fig2.jpg',x=150,y=62,w=100)
    pdf.image('fig3.jpg',x=45,y=140,w=200)
    pdf.add_page()
    pdf.cell(30, 10, txt="Connotation Analysis:", ln=1, align="C")
    pdf.image('fig4.jpg',x=120,y=12,w=100)
    pdf.image('fig5.jpg',x=-120,y=100,w=500)
    pdf.output("yourfile.pdf", "F")
    webbrowser.open_new(r'yourfile.pdf')

if __name__ == "__main__": 
    sample=str(sys.argv[1])
    main(sample)