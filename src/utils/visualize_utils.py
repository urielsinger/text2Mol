
def plot_dict(dict, ax, **kwargs):
    ax.plot(*zip(*sorted(dict.items())),**kwargs)

def scatter_dict(dict, ax, **kwargs):
    ax.scatter(*zip(*dict.items()),**kwargs)
