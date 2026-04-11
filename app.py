import streamlit as st
from rag import get_answer

st.title("📘 AI Employee Assistant")

# Role selection
role = st.selectbox("Select your role", ["Employee", "Manager", "Admin"])

# Example questions
st.write("### 📌 Example Questions:")
st.write("- What is leave policy?")
st.write("- What is dress code?")
st.write("- What is employee health policy?")

# Input
query = st.text_input("Ask your question:")

if query:
    # Authorization
    restricted_keywords = ["salary", "payroll", "confidential"]

    if role == "Employee" and any(word in query.lower() for word in restricted_keywords):
        st.error("❌ You are not authorized to access this information")
    else:
        st.success("⏳ Processing your request...")

        try:
            response = get_answer(query)

            st.write("### ✅ Answer:")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")