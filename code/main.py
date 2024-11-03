# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/3/10
import subprocess


def main():
    subprocess.call(
        "python aor.py \
            --task GSM8K \
            --data-path aor_outputs/GSM8K_CoT_gpt-3.5-turbo-0301.jsonl \
            --record-path aor_records/GSM8K_AoR_log_gpt-3.5-turbo-0301.jsonl \
            --inference-model gpt-35-turbo-0301",
        shell=True,
    )


if __name__ == "__main__":
    main()
