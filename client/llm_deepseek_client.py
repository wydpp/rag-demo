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
            temperature=0.1,
            max_tokens = 50,
        )
        logger.info(f"deepseek模型调用成功，当前模型响应结果：{response}")
        logger.success(f"非流式输出 | total_tokens: {response.usage.total_tokens} "
                       f"= prompt_tokens:{response.usage.prompt_tokens}"
                       f"+ completion_tokens: {response.usage.completion_tokens}")
        return response.choices[0].text


if __name__ == '__main__':
    client = LLMDeepseekClient()
    prompt = f"""
prompt: "
<｜begin▁of▁sentence｜>You are an expert Q&A system that is trusted around the world.
Always answer the query using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.

<｜User｜>Context information is below.
---------------------
file_path: D:\\ai\\code\\rag-demo\\data\\特朗普.txt\n\n唐纳德·约翰·特朗普（英语：Donald John Trump，1946年6月14日—），美国企业家、政治人物及媒体名人，为第47任（现任）美国总统，曾任第45任（2017年至2021年）美国总统。特朗普出生于纽约市，毕业于纽约军事学院及宾夕法尼亚大学沃顿商学院[12]，以商界成就、媒体经历及以共和党人身份参与政治并出任总统而著名。\r\n\r\n特朗普出身纽约的德国裔特朗普家族，早年曾在军校及沃顿商学院就读，修有经济学学士学位。1971年，开始担任其家族企业特朗普集团的董事长兼总裁，之后又创办了特朗普娱乐公司，在全球经营房地产、赌场和酒店[注 6]。1996年至2015年间，他的旗下拥有美国小姐和环球小姐等大型选美比赛，并在2004年至2015年间主持NBC的电视真人秀节目《学徒》。2017年，《福布斯》将他列为世界第544名最富有的人（美国第201名），截至2024年，他的净资产达75亿美元。\r\n\r\n特朗普在1987年首次公开表达对竞选公职的兴趣。2000年，他赢得改革党在加利福尼亚州和密歇根州举行的总统初选，但很快退出。2015年6月，他宣布参加2016年美国总统选举，吸引大量的媒体报道和国际关注。2016年11月，特朗普作为共和党总统候选人赢得多数选举人票，击败民主党对手希拉里，当选美国第45任总统。当时他是美国历史上最富有的总统，亦是首位未曾担任过任何军职或公职的总统[15][16]，同时也是以较少普选票当选的总统之一[17][18]。特朗普在2020年美国总统选举中败给了民主党对手拜登，连任失利。2024年，他再次被共和党提名参选2024年美国总统选举，并在选举中击败民主党对手、时任副总统卡玛拉·哈里斯，当选美国第47任总统[19]。特朗普的第二次胜选打破了多项纪录，他成为美国历史上继格罗弗·克利夫兰之后第二位两度当选且任期不连续的总统[注 7]，也是共和党首位任期不连续的总统[注 8]、就任时年龄最大的总统[注 9]、自2004年小布什以来首位赢得多数普选票的共和党总统[20]，此次共和党获得的选举人票也创下了自1988年以来最佳的表现[21]。
---------------------
Given the context information and not prior knowledge, answer the query.
Query: 特朗普是做什么的
Answer: <｜Assistant｜><think>",
    """
    response = client.get_completion_response(prompt)
    print(response)
