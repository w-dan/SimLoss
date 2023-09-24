import os
import re
import pandas as pd
from dotenv import load_dotenv
from hugchat import hugchat
from hugchat.login import Login

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
CSV_FILE_NAME = "dataset.csv"
CSV_PATH = f'../data/{CSV_FILE_NAME}'

def login_hugginchat():
    # Log in to huggingface and grant authorization to huggingchat
    sign = Login(EMAIL, PASSWORD)
    cookies = sign.login()

    # Save cookies to the local directory
    cookie_path_dir = "./cookies_snapshot"
    sign.saveCookiesToDir(cookie_path_dir)

    return cookies


def generate_abstract(abstract, cookies):
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
    return chatbot.chat(f"Paraphrase the following text: '{abstract}'")

    id = chatbot.new_conversation()
    chatbot.change_conversation(id)

    conversation_list = chatbot.get_conversation_list()

    # Switch model (default: meta-llama/Llama-2-70b-chat-hf. )
    chatbot.switch_llm(0) # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
    chatbot.switch_llm(1) # Switch to `meta-llama/Llama-2-70b-chat-hf`


def load_dataset():
    arxiv_data = pd.read_csv("../data/arxiv_data.csv")
    df = pd.DataFrame(arxiv_data)
    abstracts = df['summaries']

    return abstracts


def save_to_csv(abstract, paraphrase_abstract):
    abstract = abstract.replace('\n', '')
    print(abstract)
    print("-"*40)
    print(paraphrase_abstract)
    
    score = input('Score [0-1]: ')

    with open(CSV_PATH, 'a') as file:
        # Write the content to the file
        file.write(f"{abstract}\_._/ {paraphrase_abstract}\_._/ {score}\n")  # You can add a newline character if needed

    os.system("clear")

if __name__ == '__main__':

    # log huggingchat
    print("[+] Logging to huggingchat")
    cookies = login_hugginchat()

    # load_dataset
    print("[+] Loading abstracts")
    abstracts = load_dataset()

    # generate csv
    print("[+] CSV generated")
    with open(CSV_PATH, 'a') as file:
        # Write the content to the file
        file.write("abstract, paraphrase_abstract, score\n")  # You can add a newline character if needed

    for abstract in abstracts:
        gen = generate_abstract(abstract, cookies)
        save_to_csv(abstract, gen)
