
from cocotb.handle import BinaryValue, ModifiableObject, NonHierarchyIndexableObject
import re, os

def convert_dut_var(v):
    if type(v) == BinaryValue or type(v) == str:
        h = bits_to_hex(v)
        d = hex_fp_value_to_decimal(h)
        return v, h, d
    if type(v) == ModifiableObject:
        # need to call .value to make a BinaryValue
        return convert_dut_var(v.value)
    elif type(v) == NonHierarchyIndexableObject:
        # basically an array of values; recall on each
        return [convert_dut_var(e) for e in v.value]
    elif type(v) == list:
        # basically an array of values; recall on each
        return list(map(convert_dut_var, v))
    else:
        raise Exception("unsupported type", type(v))

def print_value_per_line(header, v):
    print(header)
    print("\n".join(map(str, enumerate(convert_dut_var(v)))))

def unpack_binary(values, W=16):
    values = str(values)
    if no_value_yet(values): return values
    assert len(values) >= W and len(values) % W == 0
    results = []
    while len(values) > 0:
        results.append(values[:W])
        values = values[W:]
    return results



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

class StateIdToStr(object):

    def __init__(self, module_fname):

        # assumes location of this file w.r.t src; clumsy :/
        fname = os.path.join(os.path.dirname(__file__), '..', 'src', module_fname)
        with open(fname, 'r') as f:
            lines = f.readlines()
        lines = list(map(str.strip, lines))

        # search forward from start for state register definition
        state_defn_re = re.compile(r"^reg \[.*?\] state.*")
        state_defn_line_num = None
        for i, line in enumerate(lines):
            if state_defn_re.match(line):
                state_defn_line_num = i
                break
        if state_defn_line_num is None:
            raise Exception("couldn't find state register definition")

        # search backwards to find localparam definition
        i = state_defn_line_num - 1
        localparam_line_num = None
        while i > 0:
            if lines[i] == 'localparam':
                localparam_line_num = i
                break
            i -= 1
        if localparam_line_num is None:
            raise Exception("couldn't find localparam defintion")

        # scan between the localparam and state_defn and
        # extract state name to state id mapping
        state_definition_re = re.compile("(.*?)=(.*)[,;]$")
        self.state_id_to_str_dict = {}
        for i in range(localparam_line_num+1, state_defn_line_num):
            line = lines[i]
            # ignore empty lines
            if len(line) == 0:
                continue
            # remve potential trailing comments
            line = re.sub("//.*", '', line).strip()

            # extract name and id
            m = state_definition_re.match(line)
            if not m:
                raise Exception(f"line [{lines[i]}] didn't match expected state definition")
            state_name = m.group(1).strip()
            state_id = int(m.group(2))
            self.state_id_to_str_dict[state_id] = state_name

        if len(self.state_id_to_str_dict) == 0:
            raise Exception("no states extracted?")

    def __getitem__(self, i):
        try:
            return f"{self.state_id_to_str_dict[int(i)]} ({i})"
        except ValueError as e:
            # xxxx value?
            return "xxxx"