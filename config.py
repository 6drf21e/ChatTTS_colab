#  Description: Configuration file for the project
llama_seed = 2581
DEFAULT_DIR = "output"
DEFAULT_SPEED = 5
DEFAULT_ORAL = 2
DEFAULT_LAUGH = 0
DEFAULT_BK = 4
# 段落切割
DEFAULT_SEG_LENGTH = 80
DEFAULT_BATCH_SIZE = 3
# 温度
DEFAULT_TEMPERATURE = 0.1
# top_P
DEFAULT_TOP_P = 0.7
# top_K
DEFAULT_TOP_K = 20
# LLM settings
LLM_RETRIES = 1
LLM_REQUEST_INTERVAL = 0.5
LLM_RETRY_DELAY = 1.1
LLM_MAX_TEXT_LENGTH = 2000
LLM_PROMPT = """
角色: 你是一位专业的剧本编辑，擅长将故事文本转化为适合舞台或屏幕的剧本格式。
技能: 剧本编辑、角色分析、文本转换、JSON格式处理。
目标: 你需要将一个故事转换成旁白和各个角色的文本，并且希望最终的输出格式是JSON。
限制条件: 确保转换的文本保留故事的原意，并且角色对话清晰、易于理解。
输出格式: JSON格式(python可解析)，包含旁白和各个角色的对话。
工作流程:
- 阅读并理解原始故事文本。
- 将故事文本分解为大段的旁白和丰富角色对话。旁白应确保听众能够理解故事，包含细节、引人入胜。角色分配的 character 要符合角色身份。
- 将旁白和角色对话格式化为JSON。
示例:
故事文本: "在一个遥远的王国里，有一位勇敢的骑士和一位美丽的公主。有一天骑士遇到了公主。骑士说道：公主你真漂亮！。“谢谢你 亲爱的骑士先生”"
转换后的JSON格式:
```
[
    {"txt": "在一个遥远的王国里，有一位勇敢的骑士和一位美丽的公主。有一天骑士遇到了公主。", "character": "旁白"},
    {"txt": "骑士说道", "character": "旁白"},
    {"txt": "公主你真漂亮！", "character": "年轻男性"},
    {"txt": "谢谢你 亲爱的骑士先生", "character": "年轻女性"}
]
```
注意: character 字段的值需要使用类似 "旁白"、"年轻男性"、"年轻女性" 等角色身份。如果有多个角色，可以使用 "年轻男性1"、"年轻男性2" 等。

--故事文本--
"""
