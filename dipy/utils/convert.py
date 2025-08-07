"""Module for different conversion functions."""


def expand_range(range_str):
    """Convert a string with ranges to a list of integers.

    Parameters
    ----------
    range_str : str
        string with ranges, e.g. '1,2,3-5,7'

    Returns
    -------
    list(int)

    """
    range_str = range_str.replace(" ", "").strip()
    range_str = range_str[:-1] if range_str.endswith(",") else range_str
    result = []
    for part in range_str.split(","):
        if not part.replace("-", "").isdigit():
            raise ValueError(f"Incorrect character found in input: {part}")
        if "-" in part:
            start, end = map(int, part.split("-"))
            result += list(range(start, end + 1))
        else:
            result.append(int(part))
    return result
