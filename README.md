````markdown
# 🧠 YouTube Transcript Summarizer Agent with LangChain + Ollama

This project extracts and summarizes YouTube video transcripts using the power of LLMs via LangChain and Ollama. It also allows users to extract key topics and important quotes from any YouTube video — all with a single query.

Built with:
- `LangChain`
- `Ollama` (e.g., `gemma3:1b`, `llama3`)
- `YouTubeTranscriptAPI`

---

## 🚀 Features

✅ Extracts transcript from any public YouTube video  
✅ Summarizes long transcripts into short, digestible content  
✅ Extracts **main topics** from the video  
✅ Pulls out **important quotes** and statements  
✅ Simple agent-based query handling using LangChain tools  
✅ Modular Python code, easy to extend

---

## 🧩 How It Works

1. Extract transcript from YouTube using `youtube_transcript_api`
2. Split the transcript into chunks using `RecursiveCharacterTextSplitter`
3. Use Ollama LLM (`gemma3:1b`) with LangChain to:
   - Summarize the content
   - Extract topics
   - Identify important quotes
4. Agent routes user queries to the right tool

---

## 📦 Installation

### 🔧 Prerequisites
- Python 3.10+
- Ollama running locally with your preferred model installed (e.g., `gemma3:1b`)
- Git

### 🛠️ Setup

```bash
git clone https://github.com/abdurrahimcs50/youtube_transcript_ai_agent.git
cd youtube-transcript-summarizer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
````

Make sure you have Ollama installed and running:

```bash
ollama run gemma3:1b
```

---

## 🧪 Usage

You can run the script directly:

```bash
python process_video.py
```

By default, it processes this video:

```
https://www.youtube.com/watch?v=1aA1WGON49E
```

The script will:

* Print the raw transcript
* Chunk it
* Run your custom query (e.g., "Can you give me a summary?")
* Print the LLM’s response

To customize the video URL or the query, modify this section in `main.py`:

```python
url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
user_query = "Can you give me a summary of this video?"
```

---

## 🧠 Example Queries

* "Give me a summary of this video."
* "What are the main topics covered?"
* "What are some important quotes?"

---

## 📁 Project Structure

```bash
.
├── main.py                  # Main script to run the summarizer
├── README.md                # You're reading it 😄
├── requirements.txt         # Dependencies
```

---

## ⚙️ Requirements

```txt
youtube-transcript-api
langchain
langchain-ollama
```

> Add other dependencies if you're using tools like `streamlit`, `gradio`, or others.

---

## 💡 Future Improvements

* Add Streamlit/Gradio UI for real-time interaction
* Allow model switching (LLaMA3, Mistral, Zephyr, etc.)
* Save results (summaries, topics, quotes) to a `.json` or `.md` file
* Optional support for multilingual transcripts

---

## 📜 License

MIT License. Use it freely in your own Gen AI experiments or freelance projects.

---

## 🤝 Contributing

PRs are welcome! You can open an issue or reach out to me on [LinkedIn](https://www.linkedin.com/in/abdurrahimcs50) if you'd like to collaborate.

---

## 🔗 Let's Connect

Made by [MD Abdur Rahim ](https://github.com/abdurrahimcs50)
Freelance Python Developer | LangChain & AI Apps Specialist

```
