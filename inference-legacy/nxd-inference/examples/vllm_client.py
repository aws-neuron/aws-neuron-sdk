from openai import OpenAI


client = OpenAI(api_key="EMPTY", base_url="http://0.0.0.0:8080/v1")
models = client.models.list()
model_name = models.data[0].id

prompt = "Hello, my name is Llama "

response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    stream=False,
    extra_body={"top_k": 50},
)

generated_text = response.choices[0].message.content
print(generated_text)
