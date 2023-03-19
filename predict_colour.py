"""Hack The MIST"""

from predict_temperature import *


def import_colours(resolution: tuple[int, int], learning: bool = True) -> np.ndarray:
    """Import colour map."""
    cols = np.product(resolution)
    colours = import_map("colours.jpg", resolution, True)
    colours = colours.reshape(cols, 3).T.astype("float")

    if learning:
        land = load_land(resolution)
        land_reversed = np.flipud(np.fliplr(land)).reshape(cols)
        land = land.reshape(cols)

        colours_reversed = np.flipud(np.fliplr(colours))
        for i in range(3):
            colours[i][land == 0] = np.nan
            colours_reversed[i][land_reversed == 0] = np.nan
        return np.c_[remove_na_cols(colours), remove_na_cols(colours_reversed)].T
    return colours.T


def get_latitude_learning(resolution: tuple[int, int] = (360, 180)) -> np.ndarray:
    """Get latitude matrix."""
    latitude = get_latitude(resolution)
    latitude_reversed = latitude.copy()

    land = load_land(resolution)
    land_reversed = np.fliplr(np.flipud(land))

    cols = np.product(resolution)
    latitude[land.reshape(cols) == 0] = np.nan
    latitude_reversed[land_reversed.reshape(cols) == 0] = np.nan

    latitude = remove_na_rows(latitude)
    latitude_reversed = remove_na_rows(latitude)
    return np.r_[latitude, latitude_reversed]


def get_inputs_colour(resolution: tuple[int, int] = (360, 180), temp_path: str = "temp_parameters",
                      prec_path: str = "prec_parameters", learning: bool = True,
                      retrograde: bool = False) -> np.ndarray:
    """Return the inputs for the ColourNet."""
    # Setting up latitude
    if learning:
        latitude = get_latitude_learning(resolution)
        inputs = get_temp_inputs(resolution, True, (False, False, True))
    elif retrograde:
        latitude = get_latitude(resolution)
        inputs = get_temp_inputs(resolution, False, (True, False, False))
    else:
        latitude = get_latitude(resolution)
        inputs = get_temp_inputs(resolution, False, (False, False, False))

    # Load neural nets
    temp = load_temperature_net(temp_path)
    prec = load_temperature_net(prec_path)

    # Apply neural nets on inputs
    temperatures = to_array(temp(to_tensor(inputs))) + temp_offset(resolution, learning, retrograde)
    precipitation = to_array(prec(to_tensor(inputs)))
    return np.c_[temperatures, precipitation, latitude]


class ColourNet(nn.Module):
    """Our cool neural network for predicting pixel colour."""

    def __init__(self) -> None:
        """Initialize the neural network."""
        super(ColourNet, self).__init__()
        self.f = nn.Linear(25, 25)
        self.g = nn.Linear(25, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass on the input x."""
        x = F.relu(self.f(x))
        x = 255 * F.sigmoid(self.g(x) / 255)
        return x

    def randomize_weights(self) -> None:
        """Randomize parameter weights."""
        torch.nn.init.xavier_uniform_(self.f.weight, gain=2.0)
        torch.nn.init.xavier_uniform_(self.g.weight, gain=2.0)


def load_temperature_net(path: str) -> TemperatureNet:
    """Return a temperature net based on the path its parameters were saved."""
    temp_data = torch.load(path)
    temp = TemperatureNet()
    temp.load_state_dict(temp_data)
    return temp


def load_colour_net(path: str) -> ColourNet:
    """Return a temperature net based on the path its parameters were saved."""
    colour_data = torch.load(path)
    colour = ColourNet()
    colour.load_state_dict(colour_data)
    return colour


def learn_colour() -> ColourNet:
    """Start the neural network's learning."""
    torch.manual_seed(29844)
    resolution = (360, 180)

    inputs = get_inputs_colour(resolution, "temp_parameters", "prec_parameters")
    target = import_colours(resolution)

    net = ColourNet()
    net.randomize_weights()
    net.train()

    inputs, target = to_tensor(inputs), to_tensor(target)
    losses = []
    boost_losses = []
    offset = 0
    offset = gradient_descent(net, inputs, target, 0.00001, 0.9, 500, offset, 5000, False,
                              losses, boost_losses)

    return net


def predict_image(net: ColourNet, inputs: Optional[np.ndarray],
                  resolution: tuple[int, int] = (360, 180)) -> np.ndarray:
    """Return a prediction of the ColourNet net on a given input."""
    if inputs is None:
        inputs = get_inputs_colour(resolution, learning=False)
    return to_array(net(to_tensor(inputs)))


def save_image(img: np.ndarray, filename: str, resolution: tuple[int, int] = (360, 180)) -> None:
    """Save a matrix of shape (n x 2n x 3) as an RGB image."""
    w, h = resolution
    img = img.reshape((h, w, 3))
    img = np.nan_to_num(img)
    i = Image.fromarray(np.uint8(img))
    i.save(filename)
