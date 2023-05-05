import streamlit as st
import streamlit.components.v1 as stc

from data import run_data
from ml import run_ml

html_temp = """
	<div style="background-color:#0A2647;padding:10px;border-radius:10px">
	<h1 style="color:#2C74B3;text-align:center;font-family:sans-serif">Diabetes Health Risk Prediction Web</h1>
	<h3 style="color:white;text-align:center;font-family:sans-serif">Kelompok 4</h3>
	</div>
    """

def main():
    #st.title("Main App")
    stc.html(html_temp)

    menu = ["Home", "Data", "Prediction"]
    st.title("Menu")
    choice = st.selectbox("", menu)
    
    if choice == "Home":
        st.write("""
			### Early Stage Diabetes Risk Predictor App
			    - Dataset ini berisikan data-data dari tanda dan gejala atas pasien diabetes baru atau yang akan menjadi pasien diabetes.
			""")
        
        st.write("""
            #### Datasource
				- https://www.kaggle.com/uciml/pima-indians-diabetes-database
                - diabetes.csv
        """)

    elif choice == "Data":
        run_data()
    else:
        run_ml()
main()