from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


def parse_url(url: str) -> str:
    """
    Extract video ID from URL.

    Args: 
        url(str): youtube video url

    Returns:
        Youtube video's video ID
    
    """
    if "youtu.be" in url:
        return url.split("/")[-1]
    if "=" in url:
        return url.split("=")[-1]

    return url

def get_text_from_video(url: str) -> str:
    """
    Get transcript text from YouTube video.

    Args:
        url(str): youtube video url

    Returns:
        Youtube video's transcripted text
    
    """
    video_id = parse_url(url)
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        transcript_text = transcript_text.replace("\n", " ").replace("'", "")
        return transcript_text
    except Exception as e:
        return f"Failed to retrieve transcript: {str(e)}"

def create_chunks(transcript_text: str) -> list:
    """
    Split transcript text into processable chunks.

    Args:
        transcript_text (str): Youtube video's transcripted text

    Returns:
        processable chunks
    
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript_text)
    return chunks

def get_summary(chunks: list) -> str:
    """
    Summarize text chunks and create a single summary.
    
    Args:
        chunks (list): processable chunks of transcriptted text

    Returns:
        A single summary for youtube video
    """
    llm = ChatOllama(model="gemma3:1b")
    template = """Text: {text}
    Goal: Summarize given text.
    Answer: """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    summaries = [chain.invoke({"text": chunk}) for chunk in chunks]
    print(f"Generated {len(summaries)} summaries.", summaries)
    
    # Better approach to combining summaries
    combined_summary = summaries
    print(f"Combined Summary: {combined_summary}")
    
    # Create final summary
    final_summary_prompt = ChatPromptTemplate.from_template(
        "Multiple summaries: {summaries}\nGoal: Create a coherent single summary.\nAnswer: "
    )
    final_summary_chain = final_summary_prompt | llm
    final_summary = final_summary_chain.invoke({"summaries": combined_summary})
    
    return final_summary

def extract_topics(chunks:list) -> list:
    """
    Extract main topics from text chunks.
    
    Args:
        chunks (list): processable chunks of transcriptted text
    
    Returns:
        Main topic list
    """
    llm = ChatOllama(model="gemma3:1b")
    template = """Text: {text}
    Goal: Extract main topics from the given text.
    Answer: List the key topics separated by commas."""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    topics_list = [chain.invoke({"text": chunk}) for chunk in chunks]

    # Combine topics from different chunks
    all_topics = set()
    for topics in topics_list:
        # Split comma-separated topics and clean whitespace
        topic_items = [t.strip() for t in topics.split(",")]
        all_topics.update(topic_items)

    # Remove empty elements
    all_topics = {topic for topic in all_topics if topic}
    
    return list(all_topics)

def extract_quotes(chunks:list) -> list:
    """
    Extract important quotes from text chunks.
    
    Args:
        chunks (list): processable chunks of transcriptted text
    
    Returns:
        important quotes list
    """
    llm = ChatOllama(model="gemma3:1b")
    template = """Text: {text}
    Goal: Extract the most important quote from this text.
    Answer: Provide the quote as plain text."""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    quotes = [chain.invoke({"text": chunk}) for chunk in chunks]
    
    # Filter duplicate or empty quotes
    unique_quotes = []
    seen_quotes = set()
    
    for quote in quotes:
        # Normalize quote (clean whitespace and compare lowercase)
        normalized = quote.strip().lower()
        if normalized and normalized not in seen_quotes:
            unique_quotes.append(quote.strip())
            seen_quotes.add(normalized)
    
    return unique_quotes

def process_user_query(query, chunks, url):
    """Select appropriate tool based on user query and generate response."""

    # llm = OllamaLLM(model="gemma3:1b")
   

    llm = ChatOllama(model="gemma3:1b")
    # llm.invoke("Sing a ballad of LangChain.")
    
    # Create wrapper functions for tools
    def get_summary_wrapper(input_str=""):
        return get_summary(chunks)
    
    def extract_topics_wrapper(input_str=""):
        return extract_topics(chunks)
    
    def extract_quotes_wrapper(input_str=""):
        return extract_quotes(chunks)
    
    # Create tools with wrapper functions
    tools = [
        Tool(
            name="get_summary",
            func=get_summary_wrapper,
            description="Provides a detailed summary of the transcript."
        ),
        Tool(
            name="extract_topics",
            func=extract_topics_wrapper,
            description="Extracts main topics from the transcript."
        ),
        Tool(
            name="extract_quotes",
            func=extract_quotes_wrapper,
            description="Extracts important quotes from the transcript."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Call agent with user query
    response = agent.invoke(input=query, handle_parsing_errors=True)

    return response


if __name__ == "__main__":
    # Initialize LLM model
    # llm = OllamaLLM(model="llama3")
    llm = ChatOllama(model="gemma3:1b")

    # Example YouTube URL
    url = "https://www.youtube.com/watch?v=1aA1WGON49E"

    transcript_text = get_text_from_video(url)
    print(transcript_text)

    chunks = create_chunks(transcript_text)
    
    print(f"Number of chunks created: {len(chunks)}")

    user_query = "Can you give me a summary of this video?"
    
    result = process_user_query(user_query, chunks, url)
    print(f"Response: {result}")
    