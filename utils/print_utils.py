import time
# 实现了一个简单的日志记录和错误处理功能，包括定义了不同类型消息的颜色编码，以及用于打印各种类型消息（日志、警告、信息、错误）的函数，
# 并在主程序中进行了简单测试
# 定义了不同类型文本的颜色编码，包括 logs（蓝色）、info（绿色）、warning（黄色）、error（红色）、bold（加粗）和 end_color（结束颜色编码，用于恢复默认颜色）
text_colors = {
    'logs': '\033[34m',
    'info': '\033[32m',
    'warning': '\033[33m',
    'error': '\033[31m',
    'bold': '\033[1m',
    'end_color': '\033[0m'
}

# 函数使用 time.strftime 函数按照指定的格式 %Y-%m-%d %H:%M:%S 获取当前时间的字符串表示，即时间戳
def get_curr_time_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# 获取当前时间戳。
# 构建错误消息的前缀，包括红色加粗的 ERROR 标识。
# 使用传入的 printer 函数（默认为 print）打印时间戳、错误前缀和错误消息。
# 再次打印时间戳、错误前缀和 Exiting!!!，表示程序即将退出。
# 最后使用 exit(-1) 以错误码 -1 退出程序。
def print_error_message(message, printer=print):
    time_stamp = get_curr_time_stamp()
    error_str = text_colors['error'] + text_colors['bold'] + 'ERROR  ' + text_colors['end_color']
    printer('{} - {} - {}'.format(time_stamp, error_str, message))
    printer('{} - {} - {}'.format(time_stamp, error_str, 'Exiting!!!'))
    exit(-1)

# 获取当前时间戳。
# 构建日志消息的前缀，包括蓝色加粗的 LOGS 标识。
# 使用 printer 函数打印时间戳、日志前缀和日志消息。
def print_log_message(message, printer=print):
    time_stamp = get_curr_time_stamp()
    log_str = text_colors['logs'] + text_colors['bold'] + 'LOGS   ' + text_colors['end_color']
    printer('{} - {} - {}'.format(time_stamp, log_str, message))

# 获取当前时间戳。
# 构建警告消息的前缀，包括黄色加粗的 WARNING 标识。
# 使用 printer 函数打印时间戳、警告前缀和警告消息。
def print_warning_message(message, printer=print):
    time_stamp = get_curr_time_stamp()
    warn_str = text_colors['warning'] + text_colors['bold'] + 'WARNING' + text_colors['end_color']
    printer('{} - {} - {}'.format(time_stamp, warn_str, message))


# 获取当前时间戳。
# 构建信息消息的前缀，包括绿色加粗的 INFO 标识。
# 使用 printer 函数打印时间戳、信息前缀和信息消息。
def print_info_message(message, printer=print):
    time_stamp = get_curr_time_stamp()
    info_str = text_colors['info'] + text_colors['bold'] + 'INFO   ' + text_colors['end_color']
    printer('{} - {} - {}'.format(time_stamp, info_str, message))


if __name__ == '__main__':
    print_log_message('Testing')
    print_warning_message('Testing')
    print_info_message('Testing')
    print_error_message('Testing')