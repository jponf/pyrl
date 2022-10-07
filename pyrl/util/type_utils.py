import enum


class StrEnum(str, enum.Enum):  # noqa: WPS600
    """
    Enum subclass to define string constants that behave like str.

    Additionally it also redefines __str__ to return the value instead of
    showing a string that also includes the Enum's part.

    Note:
        It is not called StrEnum because this type is part of the standard
        library in 3.11, which includes many more features than this simple
        implementation.
    """

    def __str__(self) -> str:
        """String representation of the str-based enum object.

        Returns:
            The value of the enum object, which is already an str.
        """
        return self.value
