import os
import re
import pandas as pd
import json
from dotenv import load_dotenv
from hugchat import hugchat
from hugchat.login import Login

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
CSV_FILE_NAME = "dataset.csv"
CSV_PATH = f'../data/{CSV_FILE_NAME}'
DELIMITATOR = "\\"

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

    # start a new conversation
    id = chatbot.new_conversation()
    chatbot.change_conversation(id)

    return chatbot.chat(f"Paraphrase the following text, but please avoid bullet points: '{abstract}'")

    id = chatbot.new_conversation()
    chatbot.change_conversation(id)

    conversation_list = chatbot.get_conversation_list()

    # Switch model (default: meta-llama/Llama-2-70b-chat-hf. )
    chatbot.switch_llm(0) # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
    chatbot.switch_llm(1) # Switch to `meta-llama/Llama-2-70b-chat-hf`


def load_dataset():
    abstracts = []
    with open("../data/arxiv-metadata-oai-snapshot.json") as f:
        for x in f:
            # Update the regular expression pattern to handle the abstract field in JSON
            pattern = r'abstract\":\"(.*?)\"'  # Updated pattern to capture the abstract
            matches = re.findall(pattern, x, re.DOTALL)

            if matches:
                # If there are multiple matches, print all of them
                for match in matches:
                    abstracts.append(match.replace("\\n", ' ').replace("\\r", '').strip())
            else:
                print("Abstract not found in the data.")

    return abstracts


def n_dataset_generated():
    gen_data = pd.read_csv(CSV_PATH, delimiter=DELIMITATOR)
    n = len(gen_data)

    return n


def save_to_csv(abstract, paraphrase_abstract):
    abstract = abstract.replace('\n', '')
    # print(abstract)
    # print("-"*40)
    # print(paraphrase_abstract)

    # score = input('Score [0-1]: ')

    with open(CSV_PATH, 'a') as file:
        # Write the content to the file
        file.write(f'"{abstract}"{DELIMITATOR}"{paraphrase_abstract}"\n')  # You can add a newline character if needed

    # os.system("clear")

if __name__ == '__main__':
    # log huggingchat
    print("[+] Logging to huggingchat")
    cookies = login_hugginchat()

    # load dataset
    print("[+] Loading abstracts")
    abstracts = load_dataset()

    # load number of datasets generated
    n = n_dataset_generated()
    print(f"[+] {n} datasets has been generated")

    # # generate csv
    print("[+] CSV generated")
    with open(CSV_PATH, 'a') as file:
        # Write the content to the file
        file.write(f"abstract{DELIMITATOR}paraphrase_abstract\n")  # You can add a newline character if needed

    print(f"[+] Total abstracts: {len(abstracts)}")
    for i, abstract in enumerate(abstracts):
        if i < n:
            print ("[*] ")
        else:
            gen = generate_abstract(abstract, cookies)
            save_to_csv(abstract, gen)
            print(f"[*] Abstract {i+1} generated")
