import streamlit as st

# Find-S Algorithm
def find_s(examples):
    hypothesis = ['0'] * len(examples[0][0])
    for example, label in examples:
        if label == 1:
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
    return hypothesis

# Streamlit UI
st.title("🧠 Find-S Algorithm")

# Define attributes
num_attributes = st.number_input("Number of attributes:", min_value=1, step=1)
attribute_names = [st.text_input(f"Attribute {i+1}", value=f"Attribute {i+1}") for i in range(num_attributes)]

# Add examples
st.subheader("Add Examples")
if 'examples' not in st.session_state:
    st.session_state.examples = []

with st.form("example_form"):
    example_values = [st.text_input(f"{attribute_names[i]}", key=f"val_{i}") for i in range(num_attributes)]
    label = st.selectbox("Label", [1, 0], format_func=lambda x: "Positive" if x == 1 else "Negative")
    if st.form_submit_button("Add Example"):
        if all(value.strip() for value in example_values):
            st.session_state.examples.append((example_values, label))
            st.success("Example added!")
        else:
            st.error("Fill all fields.")

# Show examples
if st.session_state.examples:
    st.subheader("Examples")
    for i, (example, label) in enumerate(st.session_state.examples):
        st.write(f"Example {i+1}: {example}, Label: {'Positive' if label == 1 else 'Negative'}")

# Compute hypothesis
if st.button("Compute Hypothesis"):
    if st.session_state.examples:
        hypothesis = find_s(st.session_state.examples)
        st.subheader("Hypothesis")
        st.write({attribute_names[i]: hypothesis[i] for i in range(len(hypothesis))})
    else:
        st.warning("Add at least one example.")