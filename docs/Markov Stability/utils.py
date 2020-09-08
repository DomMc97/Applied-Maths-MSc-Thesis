import ast


def str2numeric(string):
    try:
        if isinstance(string, str):
            literal = ast.literal_eval(string)
        else:
            literal = string
        if isinstance(literal, float):
            if literal.is_integer():
                return int(literal)
        return literal
    except Exception as e:
        raise Exception(f"{e} \n {string} is {type(string)} type and it is not suitable to str2numeric")


def print_human_readible_numeric(num, round_to=2):
    num = str2numeric(num)
    if abs(num) > 1:
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num = round(num / 1000.0, round_to)
        if magnitude == 0:
            return str(num)
        else:
            return '{:.{}f}{}'.format(round(num, round_to), round_to, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    else:
        return '{:.{}f}'.format(num, round_to)