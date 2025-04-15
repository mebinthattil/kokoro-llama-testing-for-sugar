from gradio_client import Client
def llm_response(query, modelname = "Qwen/Qwen2.5-Max-Demo"):
    client = Client(modelname)
    result = client.predict(
        query=query,
        history=[],
        system='''You a chatbot for an educational kids app.
            Your primary function is to teach kids how to pronounce words right.
            You must also answer their questions in a simple manner, in no more than 20-30 words.
            You must guide the learner on how to better improve their learning and remind them from time to time about trying new words and explain their meaning.''',
        api_name="/model_chat"
    )
    return result[1][0][1]


#llm_response("Hello who are you")

def response(input):
    client = Client("huggingface-projects/llama-2-13b-chat")
    result = client.predict(
            message=input,
            param_2='''You a chatbot for an educational kids app.
            Your primary function is to teach kids how to pronounce words right.
            You must also answer their questions in a simple manner, in no more than 20-30 words.
            You must guide the learner on how to better improve their learning and remind them from time to time about trying new words and explain their meaning.''',
            param_3=1024,
            param_4=0.6,
            param_5=0.9,
            param_6=50,
            param_7=1.2,
            api_name="/chat"
    )
    return(result)


#response("Hello who are you")


import requests

def ask_llama(question, model="llama3.2:1b-instruct-q2_K"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": question,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code} - {response.text}"

#print(ask_llama("How can i learn to read faster"))
