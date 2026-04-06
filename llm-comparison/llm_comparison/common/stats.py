def safe_average(values: list[float | int | None], digits: int = 4):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), digits)


def safe_ratio(numerator: float | int, denominator: float | int | None, digits: int = 4):
    if denominator in (None, 0):
        return None
    return round(numerator / denominator, digits)

