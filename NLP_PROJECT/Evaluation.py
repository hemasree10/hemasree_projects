from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
#from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
pdfreader = PdfReader("/content/Attention is all you need.pdf")
from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
chain = load_qa_chain(OpenAI(), chain_type="stuff")
from rouge import Rouge
from base import RougeEvaluator
# golden_question_answer=[
#    {"question":"What is the Transformer model","answer":"The Transformer is a model architecture that relies entirely on an attention mechanism to draw global dependencies between input and output, eschewing recurrence. It allows for more parallelization and has been shown to achieve state-of-the-art results in translation quality after being trained for as little as twelve hours on eight P100 GPUs."},
# {"question":"What is 'Scaled Dot-Product Attention'?","answer":"Scaled Dot-Product Attention is a type of attention mechanism where the weight assigned to each value is computed by taking the dot product of the query with all keys, dividing each by the square root of the dimension of the keys, and applying a softmax function to obtain the weights on the values."},
#  {"question":"How does the Transformer model improve on the limitations of RNNs?","answer":"The Transformer model addresses the limitations of RNNs by allowing for much more parallelization during training due to the self-attention mechanism, which helps in dealing with long-range dependencies more efficiently."}
# ]
golden_question_answer=[
    {"question":"give summary of document  in 50 words","answer":"This pdf provides an introduction to Technical Analysis, detailing its principles, chart types, and key assumptions. It focuses on interpreting market trends through Japanese Candlestick patterns, such as Marubozu and Spinning Tops, to predict price movements and inform trading decisions. Key takeaways conclude each section for practical application."}
    {"question":"what are some assumptions in technical analysis?", "answer":"Markets discount everything ,The ‘how’ is more important than ‘why’, Price moves in trend, History tends to repeat itself"},
    {"question":"what is harami pattern?","answer":"Harami is a two candle pattern. The first candle is usually long and the second candle has a small body. The second candle is generally opposite in colour to the first candle. On the appearance of the harami pattern a trend reversal is possible. There are two types of harami patterns – the bullish harami and the bearish harami."},
    {"question":"what is the bearish engulfing pattern?","answer":" The bearish engulfing pattern is a two candlestick pattern which appears at the top end of the trend, thus making it a bearish pattern. The thought process remains very similar to the bullish engulfing pattern, except one has to think about it from a shorting perspective"},
     {"question":"which candle stick does not give the trader a trading signal with specific entry or an exit point?","answer":"The spinning top candlestick does not give the trader a trading signal with specific entry or an exit point"},
    {"question":"What does the Hammer candlestick pattern indicate?","answer":"The Hammer candlestick pattern, which appears at the bottom of downtrends, features a short body with a long lower shadow and a very small or no upper shadow. This pattern indicates that buyers are beginning to outnumber sellers, potentially signaling a reversal or support level."},
    {"question":"What are the main principles and techniques covered in the introduction to Technical Analysis?", "answer":"Technical Analysis (TA) is a methodology for forecasting the direction of prices through the study of past market data, primarily price and volume. It visualizes the actions of market participants via stock charts, where patterns are formed that help traders identify trading opportunities. TA operates under a few core assumptions, including the idea that the market discounts everything, prices move in trends, and that history tends to repeat itself. These principles suggest that TA is best suited for identifying short-term trades, with traders often relying on various charts like bar and line charts. TA's versatility allows it to be applied across different asset classes, as long as there is historical time series data available."}
]
from rouge_metrics import RougeMetrics

for examples in golden_question_answer:
    print("-------------------------------------")
    question = examples["question"]
    print("question:",question)
    llm_response = chain.run(input_documents=docs, question=question)
    print("llm_response:", llm_response)
    golden_answer = examples["answer"]
    print("golden_response:", golden_answer)

    metrics = RougeMetrics(ground_truth=golden_answer, predicted=llm_response)()

    for key in metrics:
        print(key, metrics.get(key) )