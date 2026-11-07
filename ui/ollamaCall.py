from ollama import chat
from ollama import ChatResponse
import ollama



ollama.create(model='example', from_='llama3.2:3b', system='You are Mario from Super Mario Bros.')

response: ChatResponse = chat(
  model='example',
  messages=[
    {
      'role': 'user',
      'content': 'what is your name?',
    },
  ],
)

print(response.message.content)