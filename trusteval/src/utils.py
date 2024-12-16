
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
    color_code = ANSI_COLORS.get(color.upper(), "") if color else ""
    bold_code = "\033[1m" if bold else ""
    underline_code = "\033[4m" if underline else ""
    reset_code = ANSI_COLORS["RESET"]

    formatted_text = sep.join(map(str, args))
    if color or bold or underline:
        formatted_text = f"{bold_code}{underline_code}{color_code}{formatted_text}{reset_code}"

    builtins.print(formatted_text, sep=sep, end=end, file=file, flush=flush)

print = colored_print
