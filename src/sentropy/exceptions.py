"""Custom exceptions raised by the diversity package.

Exceptions
----------
DiversityError
    Base class for all custom diversity exceptions.
DiversityWarning
    Base class for all custom diversity warnings.
InvalidArgumentError
    Raised when invalid argument is passed to a function.
ArgumentWarning
    Used for warnings of problematic argument choices.
"""


class DiversityError(Exception):
    """Base class for all custom exceptions raised by the package.
    Catch this to handle any sentropy error without enumerating the
    concrete subclasses (e.g. InvalidArgumentError, SpectralError).
    """
    pass


class DiversityWarning(Warning):
    """Base class for all custom warnings issued by the package.
    Catch this (e.g. with warnings.filterwarnings(..., category=
    DiversityWarning)) to handle sentropy warnings as a group.
    """
    pass


class InvalidArgumentError(DiversityError):
    """Raised when an invalid argument is passed to a function or method.
    Examples include an unknown measure name (Set.subset_diversity),
    weights/items of mismatched shape or with an all-zero column in
    power_mean, and invalid backend names.
    """
    pass


class ArgumentWarning(DiversityWarning):
    """Used for warnings about problematic (but legal) argument choices.
    The computation proceeds; the warning only alerts the user that
    the arguments may not mean what they expect. Emitted for example
    when a viewpoint parameter greater than 100 is passed on the
    command line (ValidateViewpoint), since such viewpoints are
    treated as infinity.
    """
    pass

class SpectralError(InvalidArgumentError):
    """Raised when a spectral measure such as the Vendi score cannot be computed.

    Typical causes are requesting a Vendi score with a similarity
    defined by a callable or a file — spectral measures require the
    full materialized similarity matrix for the eigendecomposition —
    and a similarity-weighted abundance matrix with no positive
    eigenvalues.

    Subclasses InvalidArgumentError, so code catching that broader
    exception also catches this.
    """
    pass
