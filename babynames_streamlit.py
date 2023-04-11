import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

url = 'https://github.com/esnt/Data/raw/main/Names/popular_names.csv'

df = pd.read_csv(url)

st.title('Baby Name Count')

selected_name = st.text_input("Input name:", "Olivia")

name_df = df[df['name']==selected_name]
fig = plt.figure()
sns.lineplot(data=name_df, x='year', y='n', hue='sex')
plt.title(f'Name Trend for {selected_name} from 1910-2021')
plt.legend(loc=(1.01,0))
st.pyplot(fig)