import pickle
import streamlit as st
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import string
from nltk import ngrams
import pandas as pd
from wordcloud import WordCloud
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import base64

st.set_page_config(
   page_title="Ex-stream-ly Cool App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

def get_base64_of_bin_file(bin_file):
     with open(bin_file, 'rb') as f:
         data = f.read()
     return base64.b64encode(data).decode()

#
def set_png_as_page_bg(png_file):
     bin_str = get_base64_of_bin_file(png_file)
     page_bg_img = '''
     <style>
     .stApp {
     background-image: url("data:image/png;base64,%s");
     background-size: cover;
     }
     </style>
     ''' % bin_str

     st.markdown(page_bg_img, unsafe_allow_html=True)
     return

set_png_as_page_bg('background.jpg')

with open('data/Danh_gia.csv', 'r') as file:
    data = pd.read_csv(file)
data.fillna('trống', inplace=True)
data['ma_san_pham'] = data['ma_san_pham'].astype(int)
with open('data/processed_data.csv', 'r') as file:
    df = pd.read_csv(file)
df.fillna('trống', inplace=True)
df['ma_san_pham'] = df['ma_san_pham'].astype(int)
with open('data/San_pham.csv', 'r') as file:
    san_pham = pd.read_csv(file)

review_with_name = pd.merge(df, san_pham, how='left')
review_positive = df.loc[df['thai_do'] == 'positive']
review_negative = df.loc[df['thai_do'] == 'negative']

#LOAD EMOJICON
with open('files/emojicon.txt', 'r', encoding="utf8") as file:
  emoji_lst = file.read().split('\n')
  emoji_dict = {}
  for line in emoji_lst:
      key, value = line.split('\t')
      emoji_dict[key] = str(value)
  file.close()
#################
#LOAD TEENCODE
with open('files/teencode.txt', 'r', encoding="utf8") as file:
  teen_lst = file.read().split('\n')
  teen_dict = {}
  for line in teen_lst:
      key, value = line.split('\t')
      teen_dict[key] = str(value)
  file.close()
#################
# LOAD ADJECTIVE LIST
with open('files/tinhtu.txt', 'r', encoding="utf8") as file:
  adj_lst = file.read().split('\n')
  file.close()
#################
# LOAD VERB LIST
with open('files/dongtu.txt', 'r', encoding="utf8") as file:
  verb_lst = file.read().split('\n')
  file.close()
#################
# LOAD NOUN LIST
with open('files/danhtu.txt', 'r', encoding="utf8") as file:
  noun_lst = file.read().split('\n')
  file.close()

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], str(txt))
    
    
def process_text(text, emoji_dict, teen_dict):
    document = str(text).lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):

        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))

        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))

        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())

        new_sentence = new_sentence+ sentence + '. '
        
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        twograms = ngrams(sentence.split(), 2)
        for grams in twograms:
            word = grams[0]+"_"+grams[1]
            if (word.lower() in adj_lst) or (word.lower() in tu_phu_dinh):
                new_document += " " + word.lower()
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

tu_phu_dinh = ['không', 'chẳng', 'chả', "chưa", "khỏi"]

st.sidebar.image('logo_hasaki.jpg', use_container_width=True)

st.title("HASAKI")
st.write("## Sentiment Analysis - Review")
menu = ["Business Objective", "Build Project", "New Prediction","Product Analysis"]
choice = st.sidebar.selectbox('Menu',menu)
st.sidebar.markdown("""---""")
st.sidebar.title("Thành viên thực hiện:")
st.sidebar.markdown(
    """
    :blue[**Bùi Văn Bình**]\t:man:
    \n\n
    :blue[**Lê Thị Thanh Trúc**]\t:woman:
    """)
st.sidebar.write("""#### Giảng viên hướng dẫn:
                  Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện:
                  12/2024""")
if choice == 'Business Objective':
    st.subheader("Business Objective")
    st.write("""
    ###### Sentiment Analysis - Phân tích tình cảm
    """)
    st.write("""###### Problem/ Requirement: Ứng dụng Sentiment Analysis - phân tích tình cảm để phân tích đánh giá người dùng về sản phẩm kinh doanh của HASAKI từ đó hiểu rõ khách hàng và cải thiện chất lượng sản phẩm""")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Nội dung đánh giá")
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan','so_sao']].head(10))
    st.write("##### Data preprocessing:")

    st.dataframe(df[['ma_san_pham','noi_dung_binh_luan','thai_do']].head(10))

    st.write("##### 2. Visualize")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Số sao:")
        figure1 = sns.countplot(data=data[['so_sao']], x='so_sao')
        figure1.set_xticklabels(['1 sao','2 sao','3 sao','4 sao','5 sao'])
        plt.xlabel('Số sao')
        plt.ylabel('Số lượng')
        st.pyplot(figure1.figure)
        plt.close()
    with col2:
        st.write("Thái độ:")
        figure2 = sns.countplot(data=df, x='thai_do')
        plt.xlabel('Thái độ')
        plt.ylabel('Số lượng')
        st.pyplot(figure2.figure)
        plt.close()

    st.write("Negative wordcloud:")
    text = " ".join(review_negative["noi_dung_binh_luan"])
    wordcloud = WordCloud().generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    st.write("Positive wordcloud:")
    text = " ".join(review_positive["noi_dung_binh_luan"])
    wordcloud = WordCloud().generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    st.write("##### 3. Build RandomForest Model: ...")
    thai_do_dict = {'positive':1, 'negative':0}
    df['thai_do'] = df['thai_do'].map(thai_do_dict)
    X_train, X_test, y_train, y_test = train_test_split(df['noi_dung_binh_luan'], df['thai_do'], test_size=0.2, random_state=42, stratify=df['thai_do'])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as file:
        rf_model = pickle.load(file)
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    score_train = rf_model.score(X_train_vectorized, y_train)
    score_test = rf_model.score(X_test_vectorized, y_test)

    y_pred = rf_model.predict(X_test_vectorized)
    y_prob = rf_model.predict_proba(X_test_vectorized)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob[:, 1])

    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    fig2 = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig2.figure)
    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good for sentiment analysis.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            lines = lines[0]
            flag = True
    if type=="Input":
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True

    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            pkl_filename = "model.pkl"
            with open(pkl_filename, 'rb') as file:
                rf_model = pickle.load(file)
            vectorizer_filename = "vectorizer_model.pkl"  # Choose a filename
            with open(vectorizer_filename, 'wb') as file:
                pickle.dump(vectorizer, file)
            lines = lines.astype(str)
            lines = covert_unicode(lines)
            lines = process_text(lines, emoji_dict, teen_dict)
            lines = process_postag_thesea(lines)
            lines = normalize_repeated_characters(lines)

             #Vectorize the input
            #vectorizer = TfidfVectorizer(max_features=500) #Make sure this matches your training vectorizer
            X_new = vectorizer.fit_transform(lines)
            #Make predictions
            y_pred_new = rf_model.predict(X_new)

            #Display predictions with sentiment labels
            for i, pred in enumerate(y_pred_new):
                sentiment = "Positive" if pred == 1 else "Negative"
                st.write(f"Prediction for line {i+1}: {sentiment} ({pred})")
            st.code("New predictions (0: Positive, 1: Negative): " + str(y_pred_new))
elif choice == 'Product Analysis':
    
    prod_list = review_with_name['ten_san_pham'].unique().tolist()
    selected_id = st.selectbox("",prod_list)
    selected_prod = review_with_name[review_with_name['ten_san_pham']==selected_id]
    selected_prod['noi_dung_binh_luan'] = selected_prod['noi_dung_binh_luan'].astype(str)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f":grey[{selected_prod['ten_san_pham'].to_list()[0]}]")
        st.write(f":grey[Ma san pham: {selected_prod['ma_san_pham'].to_list()[0]}]")
        st.write(f":grey[So luong danh gia: {selected_prod.groupby('ma_san_pham')['noi_dung_binh_luan'].count().iloc[0]}]")
        st.write(f":grey[Diem trung binh: {selected_prod['diem_trung_binh'].to_list()[0]}]")
        st.write(f":grey[Positive reviews: {selected_prod[selected_prod['thai_do'] == 'positive'].shape[0]}]")
        st.write(f":grey[Negative reviews: {selected_prod[selected_prod['thai_do'] == 'negative'].shape[0]}]")
        expander = st.expander(f"Mô tả sản phẩm")
        expander.write(f":grey[{selected_prod['mo_ta'].astype(str).to_list()[0]}]")

    with col2:
        text = " ".join(selected_prod[selected_prod["thai_do"]=="positive"]["noi_dung_binh_luan"])
        wordcloud1 = WordCloud().generate(text)

        text = " ".join(selected_prod[selected_prod["thai_do"]=="negative"]["noi_dung_binh_luan"])
        wordcloud2 = WordCloud().generate(text)

        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud1, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Reviews')
        st.pyplot(plt)

        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Reviews')
        st.pyplot(plt)
