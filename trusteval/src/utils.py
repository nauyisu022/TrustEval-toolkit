
import builtins

ANSI_COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "MAGENTA": "\033[95m",
    "WHITE": "\033[97m",
    "BLACK": "\033[90m",
    "RESET": "\033[0m",  # Reset to default
}

def colored_print(*args, sep=' ', end='\n', file=None, flush=False, color=None, bold=False, underline=False):
    """
    自定义 print 函数，支持彩色输出、加粗和下划线。

    Parameters:
    - *args: 要打印的内容
    - sep: 内容之间的分隔符，默认是空格
    - end: 结尾符，默认是换行
    - file: 输出目标，默认是 sys.stdout
    - flush: 是否强制刷新输出缓冲区
    - color: 文本颜色，支持 'RED', 'GREEN', 'BLUE', 等 ANSI_COLORS 中的颜色
    - bold: 是否加粗
    - underline: 是否添加下划线
    """
    color_code = ANSI_COLORS.get(color.upper(), "") if color else ""
    bold_code = "\033[1m" if bold else ""
    underline_code = "\033[4m" if underline else ""
    reset_code = ANSI_COLORS["RESET"]

    # 格式化文本
    formatted_text = sep.join(map(str, args))
    if color or bold or underline:
        formatted_text = f"{bold_code}{underline_code}{color_code}{formatted_text}{reset_code}"

    # 调用原生 print 函数
    builtins.print(formatted_text, sep=sep, end=end, file=file, flush=flush)

# 覆盖默认 print 函数
print = colored_print

# # 示例用法
# print("This is default text")
# print("This is red text", color="RED")
# print("This is green text with bold", color="GREEN", bold=True)
# print("This is yellow text with underline", color="YELLOW", underline=True)
# print("This is blue text with bold and underline", color="BLUE", bold=True, underline=True)