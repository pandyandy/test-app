import streamlit as st

st.subheader('Hello, world!')
st.write('Hello, world!')

if st.button('Say hello'):
    st.write('Hello, my friend!')

# create a dummy app
data = {'name': 'John', 'age': 25, 'city': 'New York'}
st.dataframe(data)