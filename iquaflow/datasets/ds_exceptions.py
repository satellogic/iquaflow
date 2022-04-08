class DSNotFound(Exception):
    """Exception raised when dataset is not found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str = "Dataset has not been found") -> None:
        self.message = message
        super().__init__(self.message)


class DSAnnotationsNotFound(Exception):
    """Exception raised when dataset annotations are not found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self, message: str = "Dataset annotations have not been found"
    ) -> None:
        self.message = message
        super().__init__(self.message)
