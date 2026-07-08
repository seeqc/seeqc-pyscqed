""" The :py:mod:`pyscqed.parameter` module defines the :class:`Param` class that is used to
manipulate a single scalar parameter used in simulations and experiments.
"""
import numpy as np
import sympy as sy

from . import text2latex as t2l


class Param:
    __valid_scalar_types = [int, float, np.float64]
    def __init__(
        self,
        name: str,
        value: float | None = None,
        bounds: list[float] = [-np.inf, np.inf],
        unit_pref: float = 1.0
    ) -> None:
        """Constructor method."""
        # Ensure name is a string and it has the correct format
        if type(name) is not str:
            raise TypeError("'name' is not a string.")
        if name.find(' ') >= 0:
            raise ValueError("'name' should not have any whitespace characters.")
        # Ensure value is a float if it is not None
        if value is not None:
            if type(value) not in self.__valid_scalar_types:
                raise TypeError("'value' is not a float.")
        # Ensure bounds is a list of two floats
        if type(bounds) is not list:
            raise TypeError("'bounds' is not a list.")
        else:
            if len(bounds) != 2:
                raise TypeError("'bounds' should only have two values, %i found." % len(bounds))
            if (type(bounds[0]) is not float or type(bounds[1]) is not float) and (type(bounds[0]) is not int or type(bounds[1]) is not int):
                raise TypeError("'bounds' should contain floats.")
        # Ensure lower bound is smaller than upper bound
        if bounds[0] > bounds[1]:
            raise ValueError("Lower bound is greater than upper bound in 'bounds'.")
        # Ensure unit pref is a float and is positive
        if type(unit_pref) not in self.__valid_scalar_types:
            raise TypeError("'unit_pref' is not a float.")
        if float(unit_pref) < 0.0:
            raise ValueError("'unit_pref' is negative.")


        self.name = name
        self.symbol = sy.symbols("%s_{%s}" % (name[0], name[1:]))
        if value is not None:
            self.__value = float(value)
        else:
            self.__value = None
        self.__lower_bound = float(bounds[0])
        self.__upper_bound = float(bounds[1])
        self.__upref = float(unit_pref)
        self.name_latex = t2l.latexify_param_name(self.name)

    def getValue(self) -> float | None:
        """ Get the current value of the parameter.

        :return: The current value of the parameter.
        :rtype: int float np.float64
        """
        return self.__value

    def setValue(self, value: float) -> None:
        """ Set the value of the parameter.

        :param value: The value to set the parameter to.
        :type value: int float np.float64

        :raises Exception: If the value is not an accepted type, or it is out of bounds.

        :return: None
        """
        # Ensure value is a float
        if type(value) not in self.__valid_scalar_types:
            raise TypeError("'value' is not a float.")

        # Check bounds
        if float(value) >= self.__upper_bound:
            raise ValueError("Param %s 'value' exceeds specified upper bound." % (self.name))
        if float(value) <= self.__lower_bound:
            raise ValueError("Param %s 'value' exceeds specified lower bound." % (self.name))
        self.__value = float(value)

    def getBounds(self) -> list[float]:
        """ Get the bounds of the parameter.

        :return: The lower and upper bounds of the parameter.
        :rtype: list of two floats
        """
        return [self.__lower_bound, self.__upper_bound]

    def setBounds(self, bounds: list[float]) -> None:
        """ Set the bounds of the parameter.

        :param bounds: The lower and upper bounds of the parameter.
        :type bounds: list of two floats, or np.inf.

        :raises Exception: If the bounds are not in the correct format, or the lower bound is greater than the upper bound.

        :return: None.
        """
        # Ensure bounds is a list of two floats
        if type(bounds) is not list:
            raise TypeError("'bounds' is not a list.")
        else:
            if len(bounds) != 2:
                raise ValueError("'bounds' should only have two values, %i found." % len(bounds))
            if (type(bounds[0]) is not float or type(bounds[1]) is not float) and (type(bounds[0]) is not int or type(bounds[1]) is not int):
                raise TypeError("'bounds' should contain floats.")
        # Ensure lower bound is smaller than upper bound
        if bounds[0] > bounds[1]:
            raise ValueError("Lower bound is greater than upper bound in 'bounds'.")

        self.__lower_bound = float(bounds[0])
        self.__upper_bound = float(bounds[1])
