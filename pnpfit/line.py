def slopes(x, y):
    rise = y[1:] - y[:-1]
    run = x[1:] - x[:-1]

    return rise / run


def intercepts(x, y):
    return y[1:] - slopes(x, y) * x[1:]


def crossings(x, y, q):
    crossings = (q - intercepts(x, y)) / slopes(x, y)

    return crossings[(crossings > x[:-1]) & (crossings < x[1:])]


def interval(x, y, q, p):
    points = crossings(x, y, q)

    if len(points) == 1:
        crossing = points[0]
        if p < crossing:
            return (0, crossing)
        else:
            return (crossing, x.max())

    if len(points) % 2 != 0:
        return None

    for low, high in [points[i:i + 2] for i in range(0, len(points), 2)]:
        if p > low and p < high:
            return (low, high)
