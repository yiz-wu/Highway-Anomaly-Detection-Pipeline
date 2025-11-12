import re

def sorted_nicely( l ):
    """ 
    Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    # convert text to integers if possible
    convert = lambda text: int(text) if text.isdigit() else text

    # order of sorting: first numbers, then alphabetically.  E.g. sorted_nicely(["a2", "a11", "2a1", "a23"]) -> ["2a1", "a2", "a11", "a23"]
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)
