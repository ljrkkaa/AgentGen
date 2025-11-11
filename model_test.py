from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "DannyShaw/AgentGen-Rep-8B"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"   # 若使用 GPU，自动映射；若仅 CPU，也可改为 device_map=None
)

# 创建生成管道
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
)

# 测试生成
prompt = "You are a planning agent in a robotics domain. Next step: Action:"
output = generator(prompt, do_sample=True)[0]["generated_text"]
print(output)
