# Research Agent
My first AI Agent which researches about any topic you give using langchain and gemini with the DuckDuckGo search and wikipedia tools

# Installation and how to use
1. Make sure that you have python installed using the command ``python --version``
2. clone the git repository:  ``git clone https://github.com/ableflyer/ResearchAgent.git``
3. create a new virtualenv ``py -m venv venv`` and activate it in powershell ``.\venv\Scripts\Activate.ps1``
4. run this command to get all the modules needed: `` pip install -r requirements.txt``
5. go to https://aistudio.google.com/apikey and click on create API key to get the gemini key
6. rename the ``.env.example`` to ``.env`` and replace ``"YOUR_API_KEY_HERE"`` with your gemini api key
7. click on run or enter the following command: ``python main.py``

Congrats!! You have set up the research agent 🥳