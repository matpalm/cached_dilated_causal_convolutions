
def _bit_not(n, n_int):
    return (1 << n_int) - 1 - n

def _twos_comp_to_signed(n, n_int):
    if (1 << (n_int-1) & n) > 0:
        return -int(_bit_not(n, n_int) + 1)
    else:
        return int(n)

def no_value_yet(s):
    s = str(s)
    return 'x' in s or 'z' in s

def bits_to_hex(value):
    value = str(value)
    if no_value_yet(value): return value
    value_len = len(value)
    value = int(value, 2)
    if value_len == 16:
        return f"{value:04x}"
    elif value_len == 32:
        return f"{value:08x}"
    elif value_len == 64:
        return f"{value:16x}"
    else:
        raise Exception(f"unexpected length {value_len}")

def hex_fp_value_to_decimal(hex_fp_str):
    if no_value_yet(hex_fp_str): return hex_fp_str
    assert type(hex_fp_str) == str
    value = int(hex_fp_str, 16)
    if len(hex_fp_str) == 4:
        # single width
        integer_bits = value >> 12
        integer_value = _twos_comp_to_signed(integer_bits, n_int=4)
        fractional_bits = value & 0xFFF
        fractional_value = fractional_bits / float(2**12)
        return integer_value + fractional_value
    elif len(hex_fp_str) == 8:
        # double width
        integer_bits = value >> 24
        integer_value = _twos_comp_to_signed(integer_bits, n_int=8)
        fractional_bits = value & 0xFFFFFF
        fractional_value = fractional_bits / float(2**24)
        return integer_value + fractional_value
    else:
        raise Exception(len(hex_fp_str))

def hex_and_dec_from_bin(b):
    h = bits_to_hex(b)
    d = hex_fp_value_to_decimal(h)
    return h, d