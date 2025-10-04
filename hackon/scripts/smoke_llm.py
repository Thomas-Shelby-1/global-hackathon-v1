# Quick local test to ensure Ollama + OpenAI client are reachable
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY","ollama"))
model = os.getenv("GEN_MODEL","qwen2.5:7b-instruct")

msg = "Return JSON: {\"ok\": true, \"note\": \"hackon is alive\"}"
resp = client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content": msg}],
    temperature=0
)
print(resp.choices[0].message.content)
