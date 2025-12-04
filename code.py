# app.py
import streamlit as st
from transformers import pipeline
from typing import Any

st.set_page_config(page_title="Nifty NLP Toolbox", layout="wide")

st.title("Nifty NLP Toolbox — paraphrase / generate / summarize / sentiment / translate")
st.caption("Pick a task, enter text, tweak params, click Run. Models download once and are cached.")

# ---- Model loader (cached) ----
@st.cache_resource(show_spinner=False)
def load_pipeline(task: str, model_name: str, device: int = -1):
    """
    Load a transformers pipeline and cache it.
    device: -1 for CPU, >=0 for GPU device id
    """
    try:
        pipe = pipeline(task, model=model_name, device=device)
    except Exception as e:
        # re-raise so UI shows error
        raise RuntimeError(f"Failed to load pipeline({task}, {model_name}): {e}")
    return pipe

# ---- Sidebar: global settings ----
with st.sidebar:
    st.header("Global settings")
    device_choice = st.selectbox("Run device", options=["cpu", "gpu (device 0)"])
    device = -1 if device_choice == "cpu" else 0

    st.markdown("**Model presets (you can change per task below)**")
    default_models = {
        "text_generation": "gpt2",
        "summarization": "facebook/bart-large-cnn",
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "translation": "Helsinki-NLP/opus-mt-en-fr",
        "paraphrase": "t5-small",
    }
    # show small note about downloads
    st.info("First run will download model weights — may take time and disk space.")

# ---- UI: choose task ----
TASKS = [
    "Text generation",
    "Summarization",
    "Sentiment analysis",
    "Translation (EN → FR)",
    "Paraphrase",
]
task = st.selectbox("Select task", TASKS)

# ---- Input ----
st.subheader("Input text")
default_text = {
    "Text generation": "AI will",
    "Summarization": (
        "Artificial intelligence is reshaping every major industry. "
        "Companies are using AI to automate repetitive tasks, improve decision-making, "
        "and create entirely new business models. While the technology promises huge "
        "benefits, it also brings challenges such as job displacement, ethical concerns, "
        "and the need for stronger regulatory frameworks. The pace of AI development "
        "continues to accelerate, making it essential for governments, businesses, "
        "and individuals to adapt quickly."
    ),
    "Sentiment analysis": "I love the new update, it made everything so much faster!",
    "Translation (EN → FR)": "How are you doing today? I hope everything is fine.",
    "Paraphrase": "Machine learning is interesting and has a wide range of applications.",
}
input_text = st.text_area("Enter text here", value=default_text[task], height=180)

# ---- Task-specific params ----
st.subheader("Task parameters")

if task == "Text generation":
    model_name = st.text_input("Model (Hugging Face ID)", value=default_models["text_generation"])
    max_new_tokens = st.number_input("Max new tokens", min_value=10, max_value=2000, value=400)
    temperature = st.slider("Temperature (sampling)", 0.0, 1.5, 0.8)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9)
    repetition_penalty = st.number_input("Repetition penalty", min_value=0.5, max_value=2.0, value=1.1, step=0.1)
    num_return_sequences = st.number_input("Num return sequences", min_value=1, max_value=5, value=1)
elif task == "Summarization":
    model_name = st.text_input("Model (Hugging Face ID)", value=default_models["summarization"])
    max_length = st.number_input("Max length (tokens)", min_value=10, max_value=1024, value=140)
    min_length = st.number_input("Min length (tokens)", min_value=1, max_value=1024, value=30)
    do_sample = st.checkbox("Do sample", value=False)
    trim_to_words = st.checkbox("Trim final summary to exact N words", value=False)
    if trim_to_words:
        n_words = st.number_input("Trim to how many words?", min_value=5, max_value=1000, value=100)
elif task == "Sentiment analysis":
    model_name = st.text_input("Model (Hugging Face ID)", value=default_models["sentiment"])
elif task == "Translation (EN → FR)":
    model_name = st.text_input("Model (Hugging Face ID)", value=default_models["translation"])
elif task == "Paraphrase":
    model_name = st.text_input("Model (Hugging Face ID)", value=default_models["paraphrase"])
    paraphrase_prefix = st.text_input("Paraphrase prefix (for T5-like models)", value="paraphrase: ")
    # optional paraphrase controls
    p_num_return_sequences = st.number_input("Num paraphrases", min_value=1, max_value=5, value=1)
    p_max_length = st.number_input("Max length for paraphrase output (tokens)", min_value=10, max_value=512, value=128)

# ---- Run button ----
run = st.button("Run")

# ---- Execution area ----
if run:
    if not input_text.strip():
        st.error("Please enter input text.")
    else:
        try:
            with st.spinner("Loading model..."):
                # Map tasks to pipeline task-ids where appropriate
                if task == "Text generation":
                    hf_task = "text-generation"
                elif task == "Summarization":
                    hf_task = "summarization"
                elif task == "Sentiment analysis":
                    hf_task = "sentiment-analysis"
                elif task == "Translation (EN → FR)":
                    # generic translation pipeline works with model provided
                    hf_task = "translation"
                elif task == "Paraphrase":
                    hf_task = "text2text-generation"
                else:
                    hf_task = None

                pipe = load_pipeline(hf_task, model_name, device=device)

            st.success("Model loaded. Running...")
            # run the selected pipeline with appropriate args
            if task == "Text generation":
                outputs = pipe(
                    input_text,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=bool(temperature > 0),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    num_return_sequences=int(num_return_sequences),
                )
                # show all returned sequences
                for i, out in enumerate(outputs):
                    st.markdown(f"**Generated sequence {i+1}:**")
                    # keys vary — 'generated_text' or 'text' depending on transformers version
                    text_key = "generated_text" if "generated_text" in out else list(out.keys())[-1]
                    st.write(out[text_key])

            elif task == "Summarization":
                summary_obj = pipe(
                    input_text,
                    max_length=int(max_length),
                    min_length=int(min_length),
                    do_sample=bool(do_sample),
                )
                raw_summary = summary_obj[0].get("summary_text") or summary_obj[0].get("generated_text") or summary_obj[0].get(list(summary_obj[0].keys())[0])
                final_summary = raw_summary
                if trim_to_words:
                    words = raw_summary.split()
                    final_summary = " ".join(words[: int(n_words) ])
                st.markdown("**Summary:**")
                st.write(final_summary)
                st.caption(f"(raw summary length: {len(raw_summary.split())} words)")

            elif task == "Sentiment analysis":
                sent = pipe(input_text)
                # typical output: [{'label': 'POSITIVE', 'score': 0.9998}]
                st.markdown("**Sentiment result:**")
                st.json(sent[0])

            elif task == "Translation (EN → FR)":
                trans = pipe(input_text)
                # typical key: 'translation_text'
                translated_text = trans[0].get("translation_text") or trans[0].get("translation") or list(trans[0].values())[0]
                st.markdown("**Translation:**")
                st.write(translated_text)

            elif task == "Paraphrase":
                # many T5 paraphrase recipes use "paraphrase: <text>" prefix
                prompt = paraphrase_prefix + input_text
                paraphrases = pipe(prompt, max_length=int(p_max_length), num_return_sequences=int(p_num_return_sequences), do_sample=True, top_p=0.95, temperature=0.8)
                st.markdown("**Paraphrase outputs:**")
                for i, p in enumerate(paraphrases):
                    key = "generated_text" if "generated_text" in p else list(p.keys())[-1]
                    st.write(f"{i+1}. {p[key]}")

        except Exception as e:
            st.error(f"Error while running model: {e}")
            st.exception(e)

# ---- Footer / tips ----
st.markdown("---")
st.caption(
    "Tips: If you run into memory errors on CPU, pick smaller models (e.g., distilbert, t5-small, gpt2). "
    "For better generation quality, use larger LMs (gpt2-medium/large/xl) on a GPU."
)
