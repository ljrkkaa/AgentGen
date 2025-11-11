import time
from openai import OpenAI


def call_chatgpt_openai(ins, keys, model="gpt-4o-mini", n=1, temperature=0):
    key_index = 0

    def get_openai_completion(prompt, api_key):
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key,
            base_url="https://api.chatanywhere.tech/v1"
        )
        response = client.chat.completions.create(
            model=model,
            n=n,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=512,
            top_p=0.95,
        )
        res = [_.message.content for _ in response.choices]
        usage = response.usage
        tokens = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
        }
        return res, tokens

    success = False
    re_try_count = 5
    ans = ""
    tokens = {}
    while not success and re_try_count >= 0:
        re_try_count -= 1
        key = keys[key_index]
        try:
            ans, tokens = get_openai_completion(ins, key)
            success = True
        except Exception as e:
            key_index = (key_index + 1) % len(keys)
            print(f"[Error] {str(e)}")
            time.sleep(3)
            print(f"Retrying with next key... ({re_try_count} tries left)")
    return success, ans, tokens


# ========= 测试代码入口 =========
if __name__ == "__main__":
    # 从 key.txt 读取 key（每行一个）
    with open("src/key.txt") as f:
        keys = [line.strip() for line in f.readlines() if line.strip()]

    # 要测试的 prompt
    instruction = "你是什么模型"

    print("=== Testing OpenAI API ===")
    success, answers, tokens = call_chatgpt_openai(instruction, keys, model="gpt-4o")

    if success:
        print("\n✅ Success!")
        print(f"Answer:\n{answers[0]}\n")
        print(f"Token usage: {tokens}")
    else:
        print("❌ Request failed after retries.")
