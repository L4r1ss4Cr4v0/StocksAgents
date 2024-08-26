
import json 
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

#criado uma função para ser usada pelo CrewIA

def fetch_stock_price(ticket):
    # indica o histórico de preços de determinada ação na bolsa em um período determinado.
    stock = yf.download(ticket, start=datetime.now().replace(year=datetime.now().year - 1).strftime('%Y-%m-%d'), end=datetime.now().strftime("%Y-%m-%d"))
    return stock

#transformando a função em ferramenta
yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    # pra que serve
    description = "Fetches stocks princes for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

#importando a LLM Gemini

#cria uma variável de ambiente para pegar a chave API da Google (Gemini)
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


#criando o agente Analista de histórico das ações

stockPriceAnalyst = Agent(
    role="Senior stock price Analiyst", 
    goal="Find the {ticket} stock price and analyses trands", 
    backstory="""You are a highly experienced in analyzing the price of an 
    specific stock and make predictions about its future price.""",
    # Para vermos todo o passo a passo do agente
    verbose=True,
    llm= llm,
    #max de interações
    max_iter= 5,
    memory= True, 
    tools=[yahoo_finance_tool],
    #Caso ele possa delegar parte da sua tarefa para outros. Apenas o agente final irá ter essa função
    allow_delegation = False
)


getStockPrice = Task(
    description= "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output = "Specify the current trend stock price - up, down or sideways. eg. stock= 'APPL, price UP'",
    agent= stockPriceAnalyst
)

#importando a ferramenta de busca. Como já existe a ferramenta no DDG é mais fácil importá-la quando comparada ao yf
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


stockAnalystWrite = Agent(
    role = "Senior Stock Analyts Writer",
    goal= "Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.",
    backstory= "You're a top stock analyst known for grasping complex concepts and crafting compelling narratives. You understand macro factors, combine theories like cycle theory and fundamental analysis, and can hold multiple perspectives in your analyses.",
    llm=llm,
    max_iter = 5,
    memory=True,
    allow_delegation = True
)

#Crindo o agente Analista de Notícias
newsAnalyst = Agent(
    role= "Stock News Analyst",
    goal="Summarize market news for the {ticket} company, noting the trend (up, down, or sideways) with context. Assign a score from 0 (extreme fear) to 100 (extreme greed) for each stock asset",
    backstory="With over 10 years of experience in market trend analysis and asset tracking, you are a master analyst in traditional markets with a deep understanding of human psychology. You assess news and its sources with a healthy skepticism.",
    llm= llm,
    max_iter= 10,
    memory= True,
    tools=[search_tool],
    allow_delegation=False
)

get_news = Task(
    description= f"""Take the stock and always include BTC to it (if not request).
    Use the search tool to search each one individually. 

    The current date is {datetime.now()}.

    Compose the results into a helpfull report""",
    expected_output = """"A summary of the overall market and one sentence summary for each request asset. 
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
""",
    agent= newsAnalyst
)

writeAnalyses = Task(
    description = "Analyze the stock price trend and news report for the {ticket} company to create a brief newsletter highlighting key points. Focus on the trend, news, and fear/greed score, along with near-future considerations, and include previous analyses of the trend and news summary.",
    expected_output= """A 3-paragraph newsletter formatted in markdown for readability, including:
3 bullet points for an executive summary
Introduction that sets the context and piques interest
Main analysis with news summary and fear/greed scores
Summary highlighting key facts and future trend prediction (up, down, or sideways).""",
    agent = stockAnalystWrite,
    context = [getStockPrice, get_news]
)

crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalyses],
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

# O input para o programa rodar é o ticket (a variável esperada). Nesse caso está rodando a ação da Apple.
# results= crew.kickoff(inputs={'ticket': 'AAPL'})

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])
