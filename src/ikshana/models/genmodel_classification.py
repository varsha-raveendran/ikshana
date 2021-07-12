"""Generate Models based on Config YAML files."""

import logging
from copy import deepcopy
from typing import Any, List

import torch
from torch import nn
from torch.nn import functional

from ikshana.utils.errors.config_not_provided import ConfigNotProvided


class Model_Classification(nn.Module):
    """Generate models for classification tasks.

    Parameters
    ----------
    cfg : Any
        [description]
    input_channels : int, optional
        [description], by default 3
    number_of_classes : int, optional
        [description], by default None

    Raises
    ------
    ConfigNotProvided
        [description]
    AttributeError
        [description]
    """

    def __init__(
        self,
        cfg: Any,
        input_channels: int = 3,
        number_of_classes: int = None,
    ):
        self.number_of_classes = number_of_classes
        super(Model_Classification, self).__init__()

        if not cfg:
            logging.error(
                "Config parameter not provided to generate model object"
            )
            raise ConfigNotProvided(cfg)

        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            with open(cfg) as file:
                self.yaml = yaml.safe_load(file)

        self.input_channels = input_channels

        if not number_of_classes:
            logging.error("Number of classes not defined")
            raise AttributeError(
                f"Number of classes value:{number_of_classes}, not provided."
            )

        self.layers = parse_model(
            config_parameter=deepcopy(self.yaml),
            input_channels=self.input_channels,
        )

    def forward(self, img):
        for layer in self.layers:
            print(layer)
            img = layer(img)
        img = img.view(-1, self.number_of_classes)

        return functional.log_softmax(img, dim=1)


def parse_model(config_parameter: dict, input_channels: int) -> List[Any]:
    """Parse the config file to return the model object.

    Parameters
    ----------
    config_parameter : dict
        Config parameters in the format of dictionary.
    input_channels : int
        Input number of channels to be passed to the model.

    Returns
    -------
    nn.Sequential
        Model object generated based on the config file.
    """
    network: dict = config_parameter["net"]

    layers: list = list()
    for block_index, block in enumerate(network):
        for index, layer_args in enumerate(block):
            if len(layer_args) == 4:
                (
                    in_channels,
                    number_of_layers,
                    func,
                    (out_channels, kernel_size, stride, padding),
                ) = layer_args

                if block_index == 0 and index == 0:
                    in_channels = input_channels

                if func == "conv":
                    for _ in range(number_of_layers):
                        layers.append(
                            conv(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                            )
                        )

            elif len(layer_args) == 5:
                (
                    in_channels,
                    number_of_layers,
                    func,
                    (out_channels, kernel_size, stride, padding),
                    additional_args,
                ) = layer_args

                if block_index == 0 and index == 0:
                    in_channels = input_channels

                if func == "conv":
                    for _ in range(number_of_layers):
                        funcs: dict = dict()
                        for arg in additional_args:
                            func_name, value = arg.split(":")
                            funcs[func_name] = eval(value)

                        layers.append(
                            conv(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                **funcs,
                            )
                        )

    layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1)))

    return layers


def conv(**kwargs) -> nn.Sequential:
    """Convolution layer defined with Batch Norm and relu as activation func.

    Returns
    -------
    nn.Sequential
        Returns a sequencial layer based on the provided inputs.
    """
    return nn.Sequential(
        nn.Conv2d(**kwargs), nn.BatchNorm2d(kwargs["out_channels"]), nn.ReLU()
    )


if __name__ == "__main__":
    model = Model_Classification("./configs/cifar.yaml", 3, 10).to("cpu")
    x = torch.rand(2, 3, 32, 32).to("cpu")
    model.forward(x)
    # for i in model.parameters():
    #     print(i)
    print("-" * 100)
    print(model)
    from torchsummary import summary

    print(summary(model, input_size=(3, 32, 32), device="cpu"))
