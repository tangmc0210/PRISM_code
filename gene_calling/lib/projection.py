
# Interpolate colors to create a smoother colormap
def interpolate_colors(colors, num_colors):
    """
    Interpolate a list of colors to create a smoother gradient.
    
    :param colors: List of color tuples.
    :param num_colors: Number of colors in the new colormap.
    :return: List of interpolated color tuples.
    """
    original_indices = np.linspace(0, 1, len(colors))
    new_indices = np.linspace(0, 1, num_colors)
    interpolated_colors = []

    for i in range(3):  # For R, G, B channels
        channel = np.array([c[i] for c in colors])
        interpolated_channel = np.interp(new_indices, original_indices, channel)
        interpolated_colors.append(interpolated_channel)

    return list(zip(interpolated_colors[0], interpolated_colors[1], interpolated_colors[2]))

