#模型下载
from modelscope import snapshot_download

# model_id 模型的id
# cache_dir 缓存到本地的路径
model_dir = snapshot_download(model_id="iic/nlp_gte_sentence-embedding_chinese-large",cache_dir="D://ai/model")