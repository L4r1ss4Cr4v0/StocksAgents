
import json 
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

#criado uma função para ser usada pelo CrewIA

def fetch_stock_price(ticket):
    # indica o histórico de preços de determinada ação na bolsa em um período determinado.
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

#transformando a def em ferramenta
yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    # pra que serve
    description = "Fetches stocks princes for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

#importando a LLM da openIA

#cria uma variável de ambiente para pegar a chave API da OpenIA
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")


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
    goal="""Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with
    the news context. For each request stock asset, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.

    You're also master level analyts in the tradicional markets and have deep understanding of human psychology.

    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism. 
    You consider also the source of the news articles. 
    """,
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
    description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
""",
    expected_output= """"An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:

    - 3 bullets executive summary 
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fead/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.
""",
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
