import streamlit as st
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="AI NLP Toolkit",
    page_icon="🤖",
    layout="centered"
)

# --- Title and Description ---
st.title("🤖 All-in-One AI NLP Toolkit")
st.markdown("""
This app uses open-source models (Hugging Face Transformers) to perform various text operations.
**Select a task from the sidebar to get started!**
""")

# --- Sidebar ---
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an Operation:",
    [
        "Text Generation", 
        "Summarization", 
        "Sentiment Analysis", 
        "Translation", 
        "Paraphrasing"
    ]
)

# --- Helper Functions (Cached for Performance) ---
# We use st.cache_resource so models are loaded only once and not on every rerun.

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_translator():
    # t5-small is versatile and supports En->Fr, En->De, En->Ro
    return pipeline("translation_en_to_fr", model="t5-small")

@st.cache_resource
def load_paraphraser():
    # Using a T5 model fine-tuned for paraphrasing
    return pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# --- Main Logic ---

if option == "Text Generation":
    st.header("📝 Text Generation")
    user_input = st.text_area("Enter a prompt:", "In a world where AI helps humans...")
    
    if st.button("Generate Text"):
        if user_input.strip():
            with st.spinner("Generating..."):
                generator = load_generator()
                # max_length controls how long the output is
                result = generator(user_input, max_length=150, num_return_sequences=1)
                st.success("Generated Text:")
                st.write(result[0]['generated_text'])
        else:
            st.warning("Please enter some text!")

elif option == "Summarization":
    st.header("📑 Summarization")
    user_input = st.text_area("Enter text to summarize (Long text recommended):", height=200)
    
    if st.button("Summarize"):
        if user_input.strip():
            with st.spinner("Summarizing..."):
                summarizer = load_summarizer()
                # Determine dynamic max_length based on input length
                input_len = len(user_input.split())
                max_len = max(10, int(input_len / 2)) 
                
                result = summarizer(user_input, max_length=max_len, min_length=10, do_sample=False)
                st.success("Summary:")
                st.write(result[0]['summary_text'])
        else:
            st.warning("Please enter some text!")

elif option == "Sentiment Analysis":
    st.header("🙂/☹️ Sentiment Analysis")
    user_input = st.text_area("Enter text to analyze:", "I love using Streamlit, it makes building apps so easy!")
    
    if st.button("Check Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                classifier = load_sentiment_analyzer()
                result = classifier(user_input)
                label = result[0]['label']
                score = result[0]['score']
                
                st.success(f"Sentiment: **{label}**")
                st.info(f"Confidence Score: {score:.4f}")
        else:
            st.warning("Please enter some text!")

elif option == "Translation":
    st.header("🌐 Translation (English to French)")
    user_input = st.text_area("Enter English text:", "Hello, how are you today?")
    
    if st.button("Translate"):
        if user_input.strip():
            with st.spinner("Translating..."):
                translator = load_translator()
                result = translator(user_input)
                st.success("Translation (French):")
                st.write(result[0]['translation_text'])
        else:
            st.warning("Please enter some text!")

elif option == "Paraphrasing":
    st.header("🔄 Paraphrasing")
    user_input = st.text_area("Enter sentence to paraphrase:", "The quick brown fox jumps over the lazy dog.")
    
    if st.button("Paraphrase"):
        if user_input.strip():
            with st.spinner("Rewriting..."):
                paraphraser = load_paraphraser()
                # The model expects "paraphrase: " prefix for best results
                input_text = "paraphrase: " + user_input + " </s>"
                result = paraphraser(input_text, max_length=128, num_beams=5, early_stopping=True)
                st.success("Paraphrased Text:")
                st.write(result[0]['generated_text'])
        else:
            st.warning("Please enter some text!")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Streamlit & Hugging Face Transformers")
