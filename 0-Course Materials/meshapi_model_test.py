from openai import OpenAI

client = OpenAI(
    base_url="https://api.meshapi.ai/v1",
    api_key="Your_MeshAPI_Key_Here"
)

response = client.chat.completions.create(
    model="ai21/jamba-1-5-large-v1",
    messages=[
        {"role": "user", "content": "Who won IPL match on 02 May 2026? and who was the highest run scorer in the same match?"}
    ]
)

print(response.choices[0].message.content)