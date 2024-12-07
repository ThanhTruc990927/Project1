import pickle
import streamlit as st

import pandas as pd
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

st.sidebar.image('logo_hasaki.jpg', use_container_width=True)

st.title("HASAKI")
st.write("## Sentiment Analysis - Review")
menu = ["Business Objective", "Build Project", "New Prediction"]
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
                  Phương Khuất Thùy""")
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
    with open('data/Danh_gia.csv', 'r') as file:
        data = pd.read_csv(file)
    data.fillna('trống', inplace=True)
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan','so_sao']].head(10))
    st.write("##### Data preprocessing:")
    with open('data/processed_data.csv', 'r') as file:
        df = pd.read_csv(file)
    df.fillna('trống', inplace=True)
    st.dataframe(df[['ma_san_pham','noi_dung_binh_luan','thai_do']].head(10))

    st.write("##### 2. Visualize")
    st.write("Số sao:")
    fig0 = sns.countplot(data=data[['so_sao']], x='so_sao')
    st.pyplot(fig0.figure)
    #st.write("Thái độ:")
    #fig1 = sns.countplot(data=df[['thai_do']], x='thai_do')
    #st.pyplot(fig1.figure)

    st.write("##### 3. Build RandomForest Model: ...")
    pkl_filename = "model.pkl"
    #pkl_vectorizer = "vectorizer_model.pkl"
    # import pickle
    with open(pkl_filename, 'rb') as file:
        rf_model = pickle.load(file)
    # doc model vectorize
    #with open(pkl_vectorizer, 'rb') as file:
        #vectorizer_model = pickle.load(file)
    thai_do_dict = {'positive':1, 'negative':0}
    df['thai_do'] = df['thai_do'].map(thai_do_dict)
    X_train, X_test, y_train, y_test = train_test_split(df['noi_dung_binh_luan'], df['thai_do'], test_size=0.2, random_state=42, stratify=df['thai_do'])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
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
            vectorizer = TfidfVectorizer(max_features=500)
            x_new = vectorizer.transform(lines)
            pkl_filename = "model.pkl"
            with open(pkl_filename, 'rb') as file:
                rf_model = pickle.load(file)
            y_pred_new = rf_model.predict(x_new)
            st.code("New predictions (0: Positive, 1: Negative): " + str(y_pred_new))
