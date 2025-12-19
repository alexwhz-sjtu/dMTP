# utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

# 日志输出的格式
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
datefmt = '%Y-%m-%d %H:%M:%S'

def setup_logger(log_file='app.log', level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):

    # 创建 logger
    logger = logging.getLogger('global_logger')
    logger.setLevel(level)

    # 防止重复添加 handler（重要！避免重复输出）
    if not logger.handlers:
        # 创建一个 handler 写入文件（带轮转）
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # 可选：同时输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(console_handler)

    return logger