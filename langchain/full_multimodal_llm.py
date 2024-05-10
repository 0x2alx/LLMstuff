import streamlit as st

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def main():

    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question=""

    st.title("Multimodal chat app")
    chat_container = st.container()

    user_input = st.text_input("Type msg here:", key="user_input", on_change=set_send_input)

    send_btn = st.button("Send", key="send_btn")

    if send_btn or st.session_state.send_input:
        if st.session_state.user_question != "":

            llm_response = "This is a response..."

            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                st.chat_message("ai").write("temp answer")

if __name__ == "__main__":
    main()