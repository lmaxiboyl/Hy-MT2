from transformers import AutoModelForCausalLM, AutoTokenizer

# Выбираем легкую версию модели для теста
model_name = "Tencent-Hunyuan/Hy-MT2-1.8B"

print("Загрузка токенизатора и модели...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    trust_remote_code=True
)

# Текст для перевода
prompt = "Translate the following text to Ukrainian: Hello, how are you doing today?"
inputs = tokenizer(prompt, return_tensors="pt")

print("Генерация перевода...")
outputs = model.generate(**inputs, max_new_tokens=50)

# Вывод результата
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nРезультат:")
print(result)