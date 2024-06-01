try:
    import cn2an
except ImportError:
    print("The 'cn2an' module is not installed. Please install it using 'pip install cn2an'.")
    exit(1)

import re
import numpy as np
import wave


def save_audio(file_name, audio, rate=24000):
    """
    保存音频文件
    :param file_name:
    :param audio:
    :param rate:
    :return:
    """
    audio = (audio * 32767).astype(np.int16)

    with wave.open(file_name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


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
    chinese_punctuation_pattern = r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-]"
    text = re.sub(chinese_punctuation_pattern, ' ', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'。{2,}', '。', text)
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
    _txt = re.sub(
        r"[^\u4e00-\u9fa5，。！？、]+", "", _txt
    )

    return _txt


def convert_numbers_to_chinese(text):
    """
    将文本中的数字转换为中文数字 例如 123 -> 一百二十三
    :param text:
    :return:
    """
    return cn2an.transform(text, "an2cn")


def split_text(text, min_length=60):
    """
    将文本分割为长度不小于min_length的句子
    :param text:
    :param min_length:
    :return:
    """
    sentence_delimiters = re.compile(r'([。？！\.\n]+)')
    sentences = re.split(sentence_delimiters, text)
    # print(sentences)
    # exit()
    result = []
    current_sentence = ''
    for sentence in sentences:
        if re.match(sentence_delimiters, sentence):
            current_sentence += sentence.strip() + '。'
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
    # result = [convert_numbers_to_chinese(remove_chinese_punctuation(_.strip())) for _ in result if _.strip()]
    result = [normalize_zh(_.strip()) for _ in result if _.strip()]
    return result


def normalize_zh(text):
    # return text_normalize(remove_chinese_punctuation(text))
    return convert_numbers_to_chinese(remove_chinese_punctuation(text))


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


if __name__ == '__main__':
    txts = [
        "电影中梁朝伟扮演的陈永仁的编号27149",
        "这块黄金重达324.75克 我们班的最高总分为583分",
        "12\~23 -1.5\~2",

    ]
    for txt in txts:
        print(txt, '-->', text_normalize(txt))
        # print(txt, '-->', convert_numbers_to_chinese(txt))
