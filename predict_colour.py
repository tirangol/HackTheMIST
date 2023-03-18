"""Hack The MIST"""

from main import *


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


class ColourNet(nn.Module):
    """Our cool neural network for predicting pixel colour."""

    def __init__(self) -> None:
        """Initialize the neural network."""
        super(ColourNet, self).__init__()
        self.f = nn.Linear(25, 25)
        self.g = nn.Linear(25, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass on the input x."""
        x = F.softplus(self.f(x))
        x = self.g(x)
        return x

    def randomize_weights(self) -> None:
        """Randomize parameter weights."""
        torch.nn.init.xavier_uniform_(self.f.weight, gain=2.0)
        torch.nn.init.xavier_uniform_(self.g.weight, gain=2.0)


def learn_colour(inputs: np.ndarray) -> ColourNet:
    """Start the neural network's learning."""
    torch.manual_seed(29844)
    resolution = (360, 180)

    target = import_colours(resolution)

    net = ColourNet()
    net.randomize_weights()
    net.train()

    inputs, target = to_tensor(inputs), to_tensor(target)
    losses = []
    boost_losses = []
    offset = 0
    offset = gradient_descent(net, inputs, target, 0.001, 0.9, 500, offset, 5000, False,
                              losses, boost_losses)

    return net
