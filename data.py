import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

help_glucose = ["Pregnancies: Jumlah berapa kali hamil.",
                "Glucose: Konsentrasi glukosa plasma selama 2 jam dalam tes toleransi glukosa oral.",
                "BloodPressure: Tekanan darah diastolik(mmHg).",
                "SkinThickness: Ketebalan lipatan kulit trisep (mm).",
                "Insulin: Insulin serum 2-jam (mu U/ml).",
                "BMI: Body mass index (kg/(m)2).",
                "DiabetesPedigreeFunction: Diabetes pedigree function (fungsi yang menilai kemungkinan diabetes berdasarkan riwayat keluarga).",
                "Age: Umur (Tahun).",
                "Outcome: Class variable (0 Tidak Diabetes, 1 Diabetes).",
                ]

s = ''
for i in help_glucose: 
        s += "- " + i + "\n"

knn = KNeighborsClassifier(leaf_size = 1,
                           n_neighbors = 24,
                           p = 1)

def run_data():
    df = pd.read_csv('./dataset/diabetes.csv')

    st.subheader("Overview Data")
    st.dataframe(df)

    with st.expander("Deskripsi Data"):
        st.markdown(s)

    with st.expander("Data Correlation"):
        fig, ax = plt.subplots()
        ax.set(xlabel="", ylabel="")
        ax.xaxis.tick_top()
        sns.heatmap(df.corr(), ax=ax,annot=True,linewidth=.5)
        st.write(fig)        

    df['group'] = pd.cut(df['Age'], bins = [18, 30, 50, 99], include_lowest = True, labels = ['18-30', '30-50', '50-99'])
    df_age = df.groupby(by = 'group').mean()

    with st.expander("Tipe Data"):
        st.dataframe(df.dtypes)

    with st.expander("Summary Data"):
        st.dataframe(df.describe())

    with st.expander("Age (Mean)"):
        st.dataframe(df_age)

    with st.expander("Distribusi Outcome"):
        st.dataframe(df["Outcome"].value_counts().rename({0: "Negative", 1: "Positive"}, axis ='index'))
        count_negatif = df["Outcome"].value_counts()[0]
        count_positif = df["Outcome"].value_counts()[1]

        data = pd.DataFrame({'x': ['0', '1'],
                     'y': [count_negatif, count_positif]})

        bar_chart = alt.Chart(data).mark_bar().encode(
            x='x',
            y='y',
        )

        st.altair_chart(bar_chart, use_container_width=True)