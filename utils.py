from consts import symbol_map


def to_valid_char(character):
    return symbol_map.get(character) if symbol_map.get(character) else character

def find_actual_char(character):
    if character.isnumeric():
        return ''
    
    for key, value in symbol_map.items():
        if value == character:
            return key

    return character