"""Training module for doctest and pytest.

This module contains some example functions and classes that need to have
doctests and pytests added.
"""

# flake8: noqa D202, D107

def exponentiate_this(b: float, n: float) -> float:
    """Calculate a power
    
    Returns b to the power n.

    Args:
        b: The base
        n: The exponent

    Returns:
        A float of b to the power n.

    Examples:
        TODO: add doctests to show:
            * basic usage using rounding
            * basic usage using ellipsis
            * failure with non float inputs
    """

    if not (isinstance(b, float) and isinstance(n, float)):
        raise TypeError('Both x and y must be of type float')
    
    return b ** n


def make_full_name_string(first_name: str, last_name: str) -> str:
    """Make a full name string

    Creates a single string containing a full name.

    Args:
        first_name: The first name
        last_name: The last name
    
    Returns:
        A string for the full name.

    Examples:
        TODO: add doctests to show:
            * basic usage
            * failure with non-string inputs
    """

    if not (isinstance(first_name, str) and isinstance(last_name, str)):
        raise TypeError('Both first_name and last_name must be of type `str`')
    
    return f"{first_name} {last_name}"


class Person:
    """Describe a person.

    This class holds information about a person and provides methods to
    personalise interactions to the person.

    Args:
        first_name: The first name of the person
        last_name: The last name of the person
    
    Attributes:
        first_name: As above
        last_name: As above

    Examples:
        TODO: add doctest to show:
        * Creation of a Person instance using representation output.
        * Exception from incorrect input types.
    """

    def __init__(self, first_name: str, last_name: str) -> None:

        # Check the init arguments    
        if not (isinstance(first_name, str) and isinstance(last_name, str)):
            raise TypeError('Both first_name and last_name must be of type `str`')

        self.first_name = first_name
        self.last_name = last_name

    def __repr__(self) -> str:
        """Represent a Person instance.
        """

        # Set how an instance of Person is represented
        return f"Person: {self.first_name} {self.last_name}"

    def greet(self, language: str = 'English', formal: bool = False) -> str:
        """Create a greeting for a Person
        
        This method returns a formal or informal greeting for a Person instance
        from a small selection of languages. The greeting can either be an
        informal first name greeting or formal full name greeting.

        Args:
            language: The case insensitive language to use for the greeting,
                currently one of English, French or Japanese.
            formal: A flag to set the formality.
        
        Returns:
            A string containing the chosen greeting

        Examples:
            TODO: Add doctests to show:
            * Basic use of informal English greeting
            * Basic use of formal French greeting
            * Case insensitivity using formal Japanese greeting
            * Exception for unknown language
        """

        if not isinstance(language, str):
            raise TypeError('The language argument must be of type str')

        if not isinstance(formal, bool):
            raise TypeError('The formal argument must be of type bool')

        # Use case insensitive language
        language = language.lower()

        # Select a greeting dictionary based on language
        if language == 'english':
            greetings = {True: "Good day", False: "Hi"}
        elif language == 'french':
            greetings = {True: "Bonjour", False: "Salut"}
        elif language == 'japanese':
            greetings = {True: "こんにちは", False: "やあ"}
        else:
            raise ValueError('Unknown greeting language')
        
        # Format and return formal or informal greeting
        if formal:
            return f"{greetings[formal]} {self.first_name} {self.last_name}"
        
        return f"{greetings[formal]} {self.first_name}"
