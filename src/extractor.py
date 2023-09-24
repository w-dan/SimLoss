import os
import requests
from dotenv import load_dotenv
from hugchat import hugchat
from hugchat.login import Login

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
CSV_FILE_NAME = "dataset.csv"

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
    print(chatbot.chat(f"Paraphrase the following text: '{abstract}'"))

    id = chatbot.new_conversation()
    chatbot.change_conversation(id)

    conversation_list = chatbot.get_conversation_list()

    # Switch model (default: meta-llama/Llama-2-70b-chat-hf. )
    chatbot.switch_llm(0) # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
    chatbot.switch_llm(1) # Switch to `meta-llama/Llama-2-70b-chat-hf`


def download_dataset():
    return ['']


def save_to_csv(abstract, paraphrase_abstract):
    score = input('Score [0-1]: ')

    with open(CSV_FILE_NAME, 'a') as file:
        # Write the content to the file
        file.write(f"{abstract}, {paraphrase_abstract}, {score}\n")  # You can add a newline character if needed

if __name__ == '__main__':

    # log huggingchat
    cookies = login_hugginchat()

    # download_dataset
    abstracts = download_dataset()

    # generate csv
    with open(CSV_FILE_NAME, 'a') as file:
        # Write the content to the file
        file.write("abstract, paraphrase_abstract, score\n")  # You can add a newline character if needed

    for abstract in abstracts:
        gen = generate_abstract(abstract, cookies)
        save_to_csv (abstract, gen)
