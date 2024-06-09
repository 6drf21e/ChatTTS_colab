try:
    import cn2an
except ImportError:
    print("The 'cn2an' module is not installed. Please install it using 'pip install cn2an'.")
    exit(1)

try:
    import jieba
except ImportError:
    print("The 'jieba' module is not installed. Please install it using 'pip install jieba'.")
    exit(1)

import re
import numpy as np
import wave
import jieba.posseg as pseg


def save_audio(file_name, audio, rate=24000):
    """
    保存音频文件
    :param file_name:
    :param audio:
    :param rate:
    :return:
    """
    import os
    from config import DEFAULT_DIR
    audio = (audio * 32767).astype(np.int16)

    # 检查默认目录
    if not os.path.exists(DEFAULT_DIR):
        os.makedirs(DEFAULT_DIR)
    full_path = os.path.join(DEFAULT_DIR, file_name)
    with wave.open(full_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    return full_path


def combine_audio(wavs):
    """
    合并多段音频
    :param wavs:
    :return:
    """
    wavs = [normalize_audio(w) for w in wavs]  # 先对每段音频归一化
    combined_audio = np.concatenate(wavs, axis=1)  # 沿着时间轴合并
    return normalize_audio(combined_audio)  # 合并后再次归一化


def normalize_audio(audio):
    """
    Normalize audio array to be between -1 and 1
    :param audio: Input audio array
    :return: Normalized audio array
    """
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def combine_audio_with_crossfade(audio_arrays, crossfade_duration=0.1, rate=24000):
    """
    Combine audio arrays with crossfade to avoid clipping noise at the junctions.
    :param audio_arrays: List of audio arrays to combine
    :param crossfade_duration: Duration of the crossfade in seconds
    :param rate: Sample rate of the audio
    :return: Combined audio array
    """
    crossfade_samples = int(crossfade_duration * rate)
    combined_audio = np.array([], dtype=np.float32)

    for i in range(len(audio_arrays)):
        audio_arrays[i] = np.squeeze(audio_arrays[i])  # Ensure all arrays are 1D
        if i == 0:
            combined_audio = audio_arrays[i]  # Start with the first audio array
        else:
            # Apply crossfade between the end of the current combined audio and the start of the next array
            overlap = np.minimum(len(combined_audio), crossfade_samples)
            crossfade_end = combined_audio[-overlap:]
            crossfade_start = audio_arrays[i][:overlap]
            # Crossfade by linearly blending the audio samples
            t = np.linspace(0, 1, overlap)
            crossfaded = crossfade_end * (1 - t) + crossfade_start * t
            # Combine audio by replacing the end of the current combined audio with the crossfaded audio
            combined_audio[-overlap:] = crossfaded
            # Append the rest of the new array
            combined_audio = np.concatenate((combined_audio, audio_arrays[i][overlap:]))

    return combined_audio


def remove_chinese_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, '，', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[。，]{2,}', '。', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^，|，$', '', text)
    return text

def remove_english_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, ',', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[,\.]{2,}', '.', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^,|,$', '', text)
    return text


def text_normalize(text):
    """
    对文本进行归一化处理
    :param text:
    :return:
    """
    from zh_normalization import TextNormalizer
    # ref: https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    # print(sentences)

    _txt = ''.join(sentences)
    # 替换掉除中文之外的所有字符
    # _txt = re.sub(
    #     r"[^\u4e00-\u9fa5，。！？、]+", "", _txt
    # )

    return _txt


def convert_numbers_to_chinese(text):
    """
    将文本中的数字转换为中文数字 例如 123 -> 一百二十三
    :param text:
    :return:
    """
    return cn2an.transform(text, "an2cn")


def detect_language(sentence):
    # ref: https://github.com/2noise/ChatTTS/blob/main/ChatTTS/utils/infer_utils.py#L55
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_word_pattern = re.compile(r'\b[A-Za-z]+\b')

    chinese_chars = chinese_char_pattern.findall(sentence)
    english_words = english_word_pattern.findall(sentence)

    if len(chinese_chars) > len(english_words):
        return "zh"
    else:
        return "en"


def split_text(text, min_length=60):
    """
    将文本分割为长度不小于min_length的句子
    :param text:
    :param min_length:
    :return:
    """
    # 短句分割符号
    sentence_delimiters = re.compile(r'([。？！\.]+)')
    # 匹配多个连续的回车符 作为段落点 强制分段
    paragraph_delimiters = re.compile(r'(\s*\n\s*)+')

    paragraphs = re.split(paragraph_delimiters, text)

    result = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue  # 跳过空段落
        # 小于阈值的段落直接分开
        if len(paragraph.strip()) < min_length:
            result.append(paragraph.strip())
            continue
        print('paragraph', paragraph)
        # 大于的再计算拆分
        sentences = re.split(sentence_delimiters, paragraph)
        current_sentence = ''
        print('sentences', sentences)

        for sentence in sentences:
            if re.match(sentence_delimiters, sentence):
                current_sentence += sentence.strip() + ''
                if len(current_sentence) >= min_length:
                    result.append(current_sentence.strip())
                    current_sentence = ''
            else:
                current_sentence += sentence.strip()

        if current_sentence:
            if len(current_sentence) < min_length and len(result) > 0:
                result[-1] += current_sentence
            else:
                result.append(current_sentence)
    print("result", result)
    if detect_language(text[:1024]) == "zh":
        result = [normalize_zh(_.strip()) for _ in result if _.strip()]
    else:
        result = [normalize_en(_.strip()) for _ in result if _.strip()]
    return result


def normalize_en(text):
    from tn.english.normalizer import Normalizer
    normalizer = Normalizer()
    return remove_english_punctuation(normalizer.normalize(text))


def normalize_zh(text):
    return process_ddd(text_normalize(remove_chinese_punctuation(text)))


def batch_split(items, batch_size=5):
    """
    将items划分为大小为batch_size的批次
    :param items:
    :param batch_size:
    :return:
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# 读取 txt 文件，支持自动判断文件编码
def read_long_text(file_path):
    """
    读取长文本文件，自动判断文件编码
    :param file_path: 文件路径
    :return: 文本内容
    """
    encodings = ['utf-8', 'gbk', 'iso-8859-1', 'utf-16']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except (UnicodeDecodeError, LookupError):
            continue

    raise ValueError("无法识别文件编码")


def replace_tokens(text):
    remove_tokens = ['UNK']
    for token in remove_tokens:
        text = re.sub(r'\[' + re.escape(token) + r'\]', '', text)

    tokens = ['uv_break', 'laugh','lbreak']
    for token in tokens:
        text = re.sub(r'\[' + re.escape(token) + r'\]', f'uu{token}uu', text)
        text = text.replace('_', '')
    return text


def restore_tokens(text):
    tokens = ['uvbreak', 'laugh', 'UNK', 'lbreak']
    for token in tokens:
        text = re.sub(r'uu' + re.escape(token) + r'uu', f'[{token}]', text)
    text = text.replace('[uvbreak]', '[uv_break]')
    return text


def process_ddd(text):
    """
    处理“地”、“得” 字的使用，都替换为“的”
    依据：地、得的使用，主要是在动词和形容词前后，本方法没有严格按照语法替换，因为时常遇到用错的情况。
    另外受 jieba 分词准确率的影响，部分情况下可能会出漏掉。例如：小红帽疑惑地问
    :param text: 输入的文本
    :return: 处理后的文本
    """
    word_list = [(word, flag) for word, flag in pseg.cut(text, use_paddle=False)]
    # print(word_list)
    processed_words = []
    for i, (word, flag) in enumerate(word_list):
        if word in ["地", "得"]:
            # Check previous and next word's flag
            # prev_flag = word_list[i - 1][1] if i > 0 else None
            # next_flag = word_list[i + 1][1] if i + 1 < len(word_list) else None

            # if prev_flag in ['v', 'a'] or next_flag in ['v', 'a']:
            if flag in ['uv', 'ud']:
                processed_words.append("的")
            else:
                processed_words.append(word)
        else:
            processed_words.append(word)

    return ''.join(processed_words)


def replace_space_between_chinese(text):
    return re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)


if __name__ == '__main__':
    # txts = [
    #     "快速地跑过红色的大门",
    #     "笑得很开心，学得很好",
    #     "小红帽疑惑地问？",
    #     "大灰狼慌张地回答",
    #     "哦，这是为了更好地听你说话。",
    #     "大灰狼不耐烦地说：“为了更好地抱你。”",
    #     "他跑得很快，工作做得非常认真，这是他努力地结果。得到",
    # ]
    # for txt in txts:
    #     print(txt, '-->', process_ddd(txt))

    txts = [
        "电影中梁朝伟扮演的陈永仁的编号27149",
        "这块黄金重达324.75克 我们班的最高总分为583分",
        "12\~23 -1.5\~2",
        "居维埃·拉色别德①、杜梅里②、卡特法日③，"

    ]
    for txt in txts:
        print(txt, '-->', text_normalize(txt))
        # print(txt, '-->', convert_numbers_to_chinese(txt))
