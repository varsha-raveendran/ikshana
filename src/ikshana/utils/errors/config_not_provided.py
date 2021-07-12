"""Function to raise error if config not provided."""

from typing import Any


class ConfigNotProvided(Exception):
    """Exception to be raised if config file is not provided.

    Parameters
    ----------
    cfg : Any
        input config parameter provided by the user.
    message : str, optional
        message to be displayed if error is invoked, by default
        "Config file not provided"
    """

    def __init__(
        self, cfg: Any, message: str = "Config file not provided"
    ) -> None:
        self.cfg = cfg
        self.message = message
        super(ConfigNotProvided, self).__init__(self.message)

    def __str__(self) -> str:
        """Message provided if something goes wrong.

        Returns
        -------
        str
            Error message.
        """
        return f"Config file type: {type(self.cfg)}. Config file not provided."
