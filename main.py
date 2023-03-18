"""Hack The MIST"""

from predict_colour import *


def matrix(elevation: np.ndarray) -> np.ndarray:
    """Process the elevation matrix of shape (n, 2n)."""
    height, width = elevation.shape
    assert width == height * 2
    assert np.all(elevation >= 0) and np.all(elevation <= 1)
    elevation *= 6500  # Max possible height, in meters
    return elevation


def temperature_offset(elevation: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Return an offset temperature prediction."""
    month = np.repeat(np.arange(0., 12.).reshape(1, 12), latitude.shape, axis=0)
    lat_offset = 70 * np.cos(latitude * np.pi / 180) - 40 - 4 / (np.e ** (latitude ** 2 / 400))
    elev_offset = -19 * elevation / 3000 + 3
    month_offset = -np.repeat(latitude.reshape(latitude.shape[0], 1), 12, axis=1) * np.cos(
        np.pi * (month - 0.25) / 6) / 6

    lat_offset = np.repeat(lat_offset.reshape(1, lat_offset.shape[0]), 12, axis=0).T
    elev_offset = np.repeat(elev_offset.reshape(1, elev_offset.shape[0]), 12, axis=0).T
    return lat_offset + elev_offset + month_offset


def matrix_to_climate(elevation: np.ndarray, temp_path: str = "temp_parameters",
                      prec_path: str = "prec_parameters") -> np.ndarray:
    """Return a (__, 25) matrix of monthly temperatures/precipitation, given elevation matrix."""
    elevation = matrix(elevation)
    height, width = elevation.shape
    resolution = (width, height)

    land = elevation != 0
    latitude = get_latitude(resolution)

    # Process elevation into inputs
    raw_inputs = process_raw_data(land, elevation, False)
    inputs = process_inputs(raw_inputs, (width, height))

    # Load neural nets
    temp_data = torch.load(temp_path)
    prec_data = torch.load(prec_path)
    temp = TemperatureNet()
    prec = TemperatureNet()
    temp.load_state_dict(temp_data)
    prec.load_state_dict(prec_data)

    # Apply neural nets on inputs
    temperatures = to_array(temp(to_tensor(inputs))) + temperature_offset(elevation, latitude)
    precipitation = to_array(prec(to_tensor(inputs)))
    return np.c_[temperatures, precipitation, latitude]


def climate_to_colour(climate: np.ndarray, colour_path: str = "colour_parameters") -> np.ndarray:
    """Return a (__ x 3) image of the world map, given the climate matrix."""
    # Load neural net
    colour_data = torch.load(colour_path)
    colour = ColourNet()
    colour.load_state_dict(colour_data)

    # Apply neural net on input
    return predict_image(colour, climate)


def elevation_to_colour(elevation: np.ndarray, save_img: bool = False,
                        img_name: str = "output.png") -> np.ndarray:
    """Main function that, given an elevation matrix of size (n x 2n) where 0 indicates
    water and >0 indicates land elevation in meters, predicts climate and returns a satellite image
    RGB (n x 2n x 3) matrix of the hypothetical Earth-like planet."""
    output = climate_to_colour(matrix_to_climate(elevation))
    if save_img:
        save_image(output, img_name)
    return output
