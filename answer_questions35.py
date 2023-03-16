import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep
import os


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')


def gpt3_embedding(content, engine='text-similarity-ada-001'):  ## Ada is priced at $0.0004 / 1K tokens
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search_index(text, data, count=2):  ## return a list of DICT with roles all equal to "user" and contents are the top 20 similar input chuncks 
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])  ## full comparation of the entire JSON indexed file
        #print(score)
        scores.append({'content': i['content'], 'score': score})
    
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)  ## the orginal list of DICT from input.json
    
    new_ordered = []
    for item in ordered:
        new_item = {"role": "user", "content": item["content"]}
        new_ordered.append(new_item)
    
    return new_ordered[0:count]  ## return the first top 20 similar text chunks but in gpt3.5-turbo accepted format


def gpt35_chat_completion(prompt, engine='gpt-3.5-turbo', temp=0.0, top_p=1.0, stop=['<<END>>']): ## 0.002 usd/ 1000 tokens
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=prompt,
                temperature=temp,
                top_p=top_p,
                )
            text = response['choices'][0]['message']['content'].strip()   ## get the completion text result per the gpt3.5 response format
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt35.txt' % time()
            
            if not os.path.exists('gpt35_logs'):
                os.makedirs('gpt35_logs')
            
            with open('gpt35_logs/%s' % filename, 'w') as outfile:
                prompt_string = ""
                for item in prompt:
                    prompt_string += f"{item['role']}. {item['content']}\n-----------"
                outfile.write('Conversation:\n\n' + prompt_string + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    with open('index.json', 'r') as infile:
        data = json.load(infile)  ## data is a list of DICT load form index.json
    #print(data)
    while True:
        query = input("Enter your question here: ")
        results = search_index(query, data)  ## results is a list of DICT that can pass directly to gpt35-turbo model
        answers = list()
        
        # answer the same question for all returned chunks
        for result in results:
            conversation = [{"role": "system", "content": "You are a helpful assistant who will examine the context provided by user and answer the user's question."}]
            conversation.append(result)
            conversation.append({"role": "user", "content": query})
            prompt = conversation
            answer = gpt35_chat_completion(prompt)  ## answer need to be a string
            print('\n\n', answer)
            answers.append(answer)  ## answers is a LIST of Str
        
        # summarize the answers together
        all_answers = '\n\n'.join(answers)
        chunks = textwrap.wrap(all_answers, 10000)
        final = list()
        
        for chunk in chunks:
            conversation = [{"role": "system", "content": "You are a helpful assistant who will examine the multiple answers user provided and summarize them into one answer."}]
            conversation.append({"role": "user", "content": chunk})
            prompt = conversation
            summary = gpt35_chat_completion(prompt)
            final.append(summary)
        print('\n\n=========\n\n', '\n\n'.join(final))