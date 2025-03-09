"""
deepseek大模型接口
"""

from loguru import logger
from openai import OpenAI

from config.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEFAULT_MODEL


# deepseek大模型接口请求客户端
class LLMDeepseekClient:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = DEFAULT_MODEL

    def get_completion_response(self, prompt):
        """
        请求模型接口，获取响应结果
        :param prompt:
        :return:
        """
        logger.info(f"deepseek模型开始调用，当前模型：{self.model}")
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=0.7,
        )
        logger.info(f"deepseek模型调用成功，当前模型响应结果：{response}")
        logger.success(f"非流式输出 | total_tokens: {response.usage.total_tokens} "
                       f"= prompt_tokens:{response.usage.prompt_tokens}"
                       f"+ completion_tokens: {response.usage.completion_tokens}")
        return response.choices[0].text


if __name__ == '__main__':
    client = LLMDeepseekClient()
    prompt = f"""
    你是一位 AI 助手，负责回答用户问题。请根据提供的检索内容，优化回答，使其更加完整、准确和可读。
    
    **用户问题**: promotion 是什么?
    
    **检索到的原始内容**:
    在 RAG（Retrieval-Augmented Generation） 中，promotion 主要指对检索到的内容进行优化，以提高最终生成内容的质量。这通常包括改写、重写、补充上下文等方式，使 LLM（大语言模型）能够更好地利用这些信息进行回答。
    promotion 的关键在于对检索到的内容进行优化，以提高最终生成的回答质量。下面是一个 Python 示例，展示如何使用 promotion 来优化检索到的内容，以便让 LLM（大语言模型）生成更高质量的答案。
    
    **在给定的上下文信息中回答查询。**:
    """
    response = client.get_completion_response(prompt)
    print(response)
