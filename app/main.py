import ollama

from retrieval import retrieve
from constants import LANGUAGE_MODEL


if __name__ == '__main__':
    newline = '\n'
    input_query = input('Ask me a question: ')
    retrieved_knowledge = retrieve(input_query)

    print('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f"""You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {newline.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    """
    # print(instruction_prompt)

    stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
    )

    # print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)