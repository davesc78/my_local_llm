# !pip install bitsandbytes


import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
### Probar con microsoft/Phi-3-medium-4k-instruct


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id


# Configure 4-bit quantization with bfloat16 compute type
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Match compute type to input
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,  # Pass the config
    device_map="auto",  # Auto-detect GPU
)


pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    tokenizer=tokenizer,  # Pass the modified tokenizer
    device_map="auto",
)


messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]


outputs = pipe(
    messages,
    max_new_tokens=64,
)
print(outputs[0]["generated_text"][-1])

outputs


def ask(user_q):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Your job is to answer the question of the user",
        },
        {"role": "user", "content": user_q},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=1000,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)


ask("Hola, soy de Chile, y quiero saber como es el clima en mi país.")
ask("¿Sabes qué es ChileCompra")
ask(
    "Si yo vendo pan amasado, ¿Me conviene ofertar a una licitación de pan para la municiaplidad de Santiago?"
)
