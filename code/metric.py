# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/3/10

import re
from collections import Counter
from math import isclose


class Metric:
    def __init__(self) -> None:
        pass

    def most_common(self, lst):
        assert lst != [], "The list is empty!"
        new_lst = [i for i in lst if i != ""]
        return Counter(new_lst).most_common(1)[0][0] if new_lst != [] else ""

    def get_consistency(self, response_list: list):
        lst = self.process_pred_list(response_list)
        assert lst != [], "The list is empty!"
        new_lst = [_ for _ in lst if _ != ""]
        return Counter(new_lst).most_common(1)[0][1] if new_lst != [] else 0

    def process_pred(self, response: str) -> str:
        return response

    def process_pred_list(self, response_list: list) -> list:
        pred_list = []
        for response in response_list:
            pred = self.process_pred(response)
            pred_list.append(pred)
        return pred_list

    def cal_acc(self, pred: str, answer: str) -> int:
        return 1 if pred == answer else 0

    def get_acc(self, response_list: list, answer: str):
        pred = self.most_common(self.process_pred_list(response_list))
        return self.cal_acc(pred, answer)


# Math Reasoning
class GSM8K_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        # 提取 "the answer is" 后的部分
        pred = response.split("the answer is")[-1]
        # 移除逗号和多余的空格
        pred = pred.replace(",", "").strip()
        # 使用正则表达式匹配数字，包括整数和小数
        pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", pred)
        # 如果找到数字，取最后一个作为预测值
        pred = pred_numbers[-1] if pred_numbers else ""
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return 0
        try:
            pred_value = float(pred)
            answer_value = float(answer.replace(",", ""))
            # 使用 isclose 比较浮点数
            return 1 if isclose(pred_value, answer_value, rel_tol=1e-5) else 0
        except ValueError:
            return 0


class MultiArith_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class SingleEq_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class AddSub_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class AQuA_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        # 提取 "the answer is" 后的部分并转换为大写
        pred = response.split("the answer is")[-1].strip().upper()
        # 匹配选项 A-E
        pred_options = re.findall(r"[A-E]", pred)
        # 如果找到选项，取最后一个作为预测值
        pred = pred_options[-1] if pred_options else ""
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return 0
        # 比较时忽略大小写和多余空格
        return 1 if pred == answer.strip().upper() else 0


class SVAMP_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


# Commonsense Reasoning
class CSQA_Metric(AQuA_Metric):
    def __init__(self) -> None:
        super().__init__()


class StrategyQA_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        # 提取 "the answer is" 后的部分并转换为小写
        pred = response.split("the answer is")[-1].lower()
        # 匹配 "yes" 或 "no"
        pred_options = re.findall(r"\b(yes|no)\b", pred)
        # 如果找到答案，取最后一个作为预测值
        pred = pred_options[-1] if pred_options else ""
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return 0
        # 比较时忽略大小写和多余空格
        return 1 if pred == answer.strip().lower() else 0
