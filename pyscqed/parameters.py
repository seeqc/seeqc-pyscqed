""" The :py:mod:`pycqed.src.parameters` module defines two classes :class:`Param` and :class:`ParamCollection` that are used to manipulate scalar parameters used in simulations and experiments.
"""
import os
from typing import Any, Callable, TypeAlias
import numpy as np
import sympy as sy
import networkx as nx
import graphviz as gv
import pydot as pd
import platform

# FIXME: Make this more intelligent
if platform.system() == 'Windows':
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

from . import text2latex as t2l
from . import util

_ParameterisationData: TypeAlias = dict[str, dict[str, Any]]
_PCInternalData: TypeAlias = tuple[dict[str, "Param"], dict[str, sy.Symbol], _ParameterisationData, nx.DiGraph]

class Param:
    """This class defines the properties of a parameter used in simulations and experiments. It supports upper and lower bounds and the generation of sweeps for scalars.
    
    :param name: The utf name of the parameter.
    :type name: str
    
    :param value: The initial value to set the parameter to, defaults to `None`.
    :type value: int float np.float64, optional
    
    :param bounds: The lower and upper bounds of the parameter, defaults to `[-np.inf, np.inf]`.
    :type bounds: list of floats, optional
    
    :param unit_pref: A prefactor that encode the unit of the parameter, defaults to `1.0`.
    :type unit_pref: int float np.float64, optional
    
    :raises Exception: If the parameters are not accepted types or the initial value is out of bounds.
    
    :return: A new instance of :class:`Param`.
    :rtype: :class:`Param`
    
    This constructor attempts to generate a latex version of the parameter name for use with plots. It uses the :py:mod:`pycqed.src.text2latex` module for this functionality. The constructor also creates a `sympy` symbol associated with the parameter. The first character is the symbol and the following characters are subscripted.
    
    The following class attributes are accessible by the user:
    
    :ivar name: The utf string of the parameter.
    :ivar name_latex: A string to be used with latex.
    :ivar symbol: A `sympy` symbol for the parameter.
    :ivar sweep: The sweep array.
    """
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
        self.sweep = np.array([])
        self.N = 0
    
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
    
    def linearSweep(self, start: float, end: float, N: int) -> np.ndarray:
        """ Generates a linear sweep using `numpy.linspace` with added bounds checking. The sweep is saved internally, and is overwritten by subsequent calls to this function.
        
        :param start: The initial value of the sweep.
        :type start: float
        
        :param end: The last value of the sweep.
        :type end: float
        
        :param N: The number of points from start to end.
        :type N: int
        
        :raises Exception: If the argument types are incorrect, or if start and end are out of bounds.
        
        :return: The parameter sweep array.
        :rtype: numpy.ndarray
        """
        # Ensure start is a float
        if type(start) not in self.__valid_scalar_types:
            raise TypeError("'start' is not a float.")
        # Ensure end is a float
        if type(end) not in self.__valid_scalar_types:
            raise TypeError("'end' is not a float.")
        # Ensure N is an int
        if type(N) is not int:
            raise TypeError("'N' is not an int.")
        
        # Check bounds
        if float(start) > self.__upper_bound:
            raise ValueError("Param %s 'start' exceeds specified upper bound." % (self.name))
        if float(start) < self.__lower_bound:
            raise ValueError("Param %s 'start' exceeds specified lower bound." % (self.name))
        if float(end) > self.__upper_bound:
            raise ValueError("Param %s 'end' exceeds specified upper bound." % (self.name))
        if float(end) < self.__lower_bound:
            raise ValueError("Param %s 'end' exceeds specified lower bound." % (self.name))
        
        self.sweep = np.linspace(float(start), float(end), N)
        self.N = N
        return self.sweep

# FIXME: This class should technically inherit the unit system
class ParamCollection:
    """ This class uses an array of :class:`Param` instances and provides methods to manipulate them in useful ways, for example to create multidimensional sweeps and return substitution dictionaries. It also provides an equation system, to allow parameters to be created in terms of others, or to specify inter-dependencies.
    
    :param names: A list of parameter names to create.
    :type names: list of str
    
    :raises Exception: If the names are not strings (raised by the underlying :class:`Param` constructor).
    
    :return: A new instance of :class:`ParamCollection`
    :rtype: :class:`ParamCollection`
    
    The following class attributes are accessible by the user. These attributes are created or overwritten by calls to :func:`ndSweep`, except the last, which is created or overwritten by calls to :func:`computeFuncSweep` and :func:`computeExprSweep`.
    
    :ivar sweep_spec: An array of sweep specifications.
    :ivar sweep_grid_npts: The number of points in the current parameters sweep.
    :ivar sweep_grid_ndims: The number of dimensions or parameters being swept.
    :ivar sweep_grid_params: The list of parameters being swept.
    :ivar sweep_grid_c: Dictionary of collapsed sweeps keyed by parameter name.
    :ivar sweep_grid_nc: Dictionary of non-collapsed sweeps keyed by parameter name.
    :ivar sweep_grid_result: One-dimensional array of values (or objects) that result from computing a sweep with the collapsed grid.
    
    """
    
    def __init__(self, names: list[str]) -> None:
        self.__collection = {}
        self.__symbol_map = {}
        self.__parameterisation = {}
        self.__parameterisation_graph = nx.DiGraph()
        for name in names:
            self.__collection[name] = Param(name)
            self.__symbol_map[name] = self.__collection[name].symbol
            self.__parameterisation_graph.add_node(name)
    
    ###################################################################################################################
    #       Basic Parameter Manipulation Functions
    ###################################################################################################################
    
    # FIXME: Should be renamed to getParameterDict
    def getParameterList(self) -> dict[str, Param]:
        """ Gets the parameter dictionary, mapping the utf name to the :class:`Param` instance.
        
        :return: A dictionary of param names to :class:`Param` instances.
        :rtype: dict
        """
        return self.__collection
    
    # FIXME: Should be renamed to getSymbolDict
    def getSymbolList(self) -> dict[str, sy.Symbol]:
        """ Gets the symbol dictionary, mapping the utf name to the Param `sympy` symbol.
        
        :return: A dictionary of param names mapping to `sympy` symbols.
        :rtype: dict
        """
        return self.__symbol_map
    
    def getSymbol(self, name: str) -> sy.Symbol:
        """ Gets the symbol associated with the parameter `name`.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The symbol of the specified parameter.
        :rtype: sympy.Symbol
        """
        if type(name) != str:
            raise TypeError("Parameter name %s is not a string." % repr(name))
        return self.__symbol_map[name]
    
    def getParameterValuesDict(self) -> dict[str, float | None]:
        """ Returns a dictionary of all the current set values of the parameters in a dictionary format.
        
        :return: A dictionary of all param names to values.
        :rtype: dict
        """
        return {k: v.getValue() for k, v in self.__collection.items()}
    
    def getSymbolValuesDict(self) -> dict[sy.Symbol, float | None]:
        """ Returns a dictionary of all the current set values of the parameter symbols in a dictionary format.
        
        :return: A dictionary of all param symbols to values.
        :rtype: dict
        """
        return {self.__symbol_map[k]: v.getValue() for k, v in self.__collection.items()}
    
    def addParameter(self, name: str, symbol_override: sy.Symbol | None = None) -> None:
        """ Adds a new parameter to the collection if it does not already exist. If it does exist, nothing is reported.
        
        :param name: The name of the parameter to add.
        :type name: str
        
        :param symbol_override: An different symbol to use than the internally generated one.
        :type symbol_override: sympy.Symbol
        
        :raises Exception: If `name` is not a str or `symbol_override` is not a sympy.Symbol.
        
        :return: None
        """
        if type(name) != str:
            raise TypeError("Parameter name %s is not a string." % repr(name))
        
        if symbol_override is not None:
            if type(symbol_override) != sy.Symbol:
                raise TypeError("Symbol %s is not a sympy.Symbol instance." % repr(symbol_override))
        
        if name not in list(self.__collection.keys()):
            self.__collection[name] = Param(name)
            if symbol_override is not None:
                self.__collection[name].symbol = symbol_override
            self.__symbol_map[name] = self.__collection[name].symbol
            self.__parameterisation_graph.add_node(name)
    
    def addParameters(self, *names: str) -> None:
        r""" Adds multiple new parameters to the collection if they do not already exist. If some or all exist, nothing is reported.
        
        :param \*name: Arguments list of parameter names to add.
        :type \*name: str, str ...
        
        :return: None
        """
        for name in list(names):
            self.addParameter(name)
    
    def rmParameter(self, name: str) -> None:
        """ Removes a parameter from the collection if it exists.
        
        :param name: The parameter to remove.
        :type name: str
        
        :raises Exception: If the parameter does not exist in the collection.
        
        :return: None
        """
        if name not in list(self.__collection.keys()):
            raise ValueError("'%s' parameter was not found." % name)
        else:
            del self.__collection[name]
            del self.__symbol_map[name]
    
    def getParameterNamesList(self) -> list[str]:
        """ Gets the list of available parameter names in the collection.
        
        :return: The list of parameter names.
        :rtype: list
        """
        return list(self.__collection.keys())
    
    def getParameterSymbolsList(self) -> list[sy.Symbol]:
        """ Gets the list of available parameter symbols in the collection.
        
        :return: The list of `sympy` symbols.
        :rtype: list
        """
        return list(self.__symbol_map.values())
    
    def getParameterFromSymbol(self, symbol: sy.Symbol) -> str | None:
        """ Gets parameter string name associated with the provided `sympy.Symbol`.
        
        :raises Exception: If the symbol does not exist in the collection.
        
        :return: The parameter name.
        :rtype: str
        """
        if symbol not in self.__symbol_map.values():
            raise ValueError("Symbol '%s' was not found." % repr(symbol))
        for k, v in self.__symbol_map.items():
            if symbol == v:
                return k
    
    def setParameterValue(self, name: str, value: float) -> None:
        """ Set the value of a given parameter.
        
        :param name: The name of the parameter to set.
        :type name: str
        
        :param value: The value to set the parameter to.
        :type value: float
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: None
        """
        # Check name is defined
        if name not in self.__collection.keys():
            raise ValueError("'%s' parameter was not found." % name)
        
        # Don't update the value if this parameter is parameterised by others
        if name in self.__parameterisation.keys():
            print("Warning: Parameter %s is parameterised so it will not be set to the requested value." % name)
            return
        
        # Update the value
        self.__collection[name].setValue(value)
        self.updateParameterisations()
    
    def getParameterValue(self, name: str) -> float | None:
        """ Get the value of a given parameter.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The current value of the specified parameter.
        :rtype: float
        """
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise ValueError("'%s' parameter was not found." % name)
        
        return self.__collection[name].getValue()
    
    def getParameterSweep(self, name: str) -> np.ndarray:
        """ Get the parameter sweep associated with a parameter.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The sweep array.
        :rtype: numpy.ndarray
        """
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise ValueError("'%s' parameter was not found." % name)
        return self.__collection[name].sweep
    
    def getParameterLatexName(self, name: str) -> str:
        """ Get the latex name of the parameter.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The parameter latex name.
        :rtype: str
        """
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise ValueError("'%s' parameter was not found." % name)
        return self.__collection[name].name_latex
    
    def setParameterValues(self, *name_value_pairs: str | float | dict[str, float]) -> None:
        r""" Set many parameter values.
        
        :param \*name_value_pairs: Arguments list, formatted as the parameter name followed by its value, or optionally passed as a dictionary.
        :type \*name_value_pairs: str, float, str, float ..., or a dict.
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: None
        """
        if len(list(name_value_pairs)) == 1:
            if type(name_value_pairs[0]) is not dict:
                raise TypeError("Argument should be a dictionary, not '%s'." % repr(type(name_value_pairs[0])))
            #for k,v in name_value_pairs[0].items():
            #    self.__collection[k].setValue(v)
            #return
        
        # Separate names from values
        keys = None
        values = None
        if type(name_value_pairs[0]) == dict:
            keys = list(name_value_pairs[0].keys())
            values = list(name_value_pairs[0].values())
        else:
            keys = list(name_value_pairs)[::2]
            values = list(name_value_pairs)[1::2]
        
        # Check there are as many parameters as values
        if len(keys) != len(values):
            raise ValueError("'name_value_pairs' definition invalid.")
        
        for i, name in enumerate(keys):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise ValueError("'%s' parameter was not found." % name)
            if name in self.__parameterisation.keys():
                print("Warning: Parameter %s is parameterised so it will not be set to the requested value." % name)
                continue
            self.__collection[name].setValue(values[i])
        self.updateParameterisations()
    
    def getParameterValues(self, *names: str) -> dict[str, float | None]:
        r""" Get the values of many parameters as a dictionary.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A dictionary with the parameter names as keys
        :rtype: dict
        """
        values = {}
        for name in list(names):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise ValueError("'%s' parameter was not found." % name)
            values[name] = self.__collection[name].getValue()
        return values
    
    def getSymbolValues(self, *names: str) -> dict[sy.Symbol, float | None]:
        r""" Get the values of many parameters keyed by symbol. Useful for getting a substitution dict of a selection of parameters.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A dictionary with the parameter symbols as keys
        :rtype: dict
        """
        values = {}
        for name in list(names):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise ValueError("'%s' parameter was not found." % name)
            values[self.__symbol_map[name]] = self.__collection[name].getValue()
        return values
    
    def allParametersSet(self) -> bool:
        """ Checks if all the parameters in the collection have been initialised.
        
        :return: True if all parameters have been initialised else False
        :rtype: bool
        """
        if None in self.getParameterValuesDict().values():
            return False
        return True
    
    ###################################################################################################################
    #       Parameterisations
    ###################################################################################################################
    
    def getSymbols(self, *names: str, symbol_overrides: list[sy.Symbol | None] | None = None) -> dict[str, sy.Symbol]:
        r""" Generates a set of `sympy` symbols for use in parameterisation. They are added as independent Param instances.
        The symbols are returned in a dictionary so that they can be used to create expressions.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A dictionary of `sympy` symbols with the parameter names as keys
        :rtype: dict
        """
        if symbol_overrides is None:
            symbol_overrides = [None]*len(list(names))
        else:
            symbol_overrides = list(symbol_overrides)
        values = {}
        for i, name in enumerate(list(names)):
            # Check name is defined
            if type(name) != str:
                raise TypeError("Parameter name %s is not a string." % repr(name))
            self.addParameter(name, symbol_override=symbol_overrides[i])
            values[name] = self.__symbol_map[name]
        return values
    
    def getParametricParametersList(self) -> list[str]:
        """ Get the list of parameters that have a parametric expression.
        
        :return: A list of parameter names.
        :rtype: list
        """
        return list(self.__parameterisation.keys())

    def addParameterisation(self, name: str, expression: sy.Expr) -> None:
        """ Registers a parameterisation of the parameter `name` in terms of symbols returned by :func:`getSymbols`. It is allowed to use `sympy` functions such as `sympy.cos` in expressions, and also any previously defined parameters. If the parameterisation already exists, it is overwritten.

        :param name: The name of the parameter that is being parameterised.
        :type name: str

        :param expression: A `sympy` expression in terms of other parameters.
        :type expression: float, int, variable
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds. Also raises an exception if this parameterisation would cause a cycle in the dependency tree of nested parameterisations.

        :return: None
        :rtype: None
        """
        if name not in list(self.__collection.keys()):
            raise ValueError("'%s' parameter was not found." % name)

        # Check the symbols in expression are actually all registered and get their names
        names = []
        rev_map = {v: k for k, v in self.__symbol_map.items()}
        for sym in expression.free_symbols:
            if sym not in self.getSymbolList().values():
                raise ValueError("Symbol '%s' not registered." % repr(sym))
            names.append(rev_map[sym])

        # Make temp copy of graph
        graph = self.__parameterisation_graph.copy()

        # Make edges from the names of included parameters
        for pname in names:
            graph.add_edge(pname, name)

        # Test that the graph is a DAG
        assert nx.is_directed_acyclic_graph(graph), "A cyclic dependency was detected with this parameterisation."

        # Register the parameterisation
        self.__parameterisation[name] = {
            "expression": expression,
            "parameters": names
        }
        self.__parameterisation_graph = graph

    def addParameterisationPrefactor(self, name: str, prefactor: sy.Expr | float) -> None:
        """ Add a prefactor to a parameterisation expression. This is mechanism for implementing unit conversion between parameters if required.
        
        :param name: The name of the parametric parameter.
        :type name: str
        
        :param prefactor: A `sympy` expression in terms of other parameters.
        :type prefactor: float, int, variable
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: None
        :rtype: None
        """
        if name not in list(self.__parameterisation.keys()):
            raise ValueError("'%s' parameter is not parameterised." % name)
        self.__parameterisation[name]["expression"] *= prefactor
    
    def getParametricExpression(self, name: str, expand: bool = False) -> sy.Expr:
        """ Gets the `sympy` expression of parameter `name`.
        
        :param name: The name of the parameter.
        :type name: str
        
        :param expand: Indicates whether to expand into nested parameterisations
        :type expand: bool, optional
        
        :raises Exception: If the parameter was not parameterised.
        
        :return: A `sympy` expression.
        :rtype: sympy type
        """
        if name not in list(self.__parameterisation.keys()):
            raise ValueError("'%s' parameter is not parameterised." % name)
        if not expand:
            return self.__parameterisation[name]["expression"]
        
        def exprParametric(expr: sy.Expr) -> bool:
            pp = self.getParametricParametersList()
            for sym in list(expr.free_symbols):
                p = self.getParameterFromSymbol(sym)
                if p in pp:
                    return True
            return False
        
        # Recursively substitute expressions
        base = self.__parameterisation[name]["expression"]
        expr = None
        while exprParametric(base):
            expr = None
            for sym in base.free_symbols:
                try:
                    local_name = self.getParameterFromSymbol(sym)                        
                    subs = {sym: self.getParametricExpression(local_name)}
                    expr = base.subs(subs)
                except: # Fail should only be caused when parameter is not parameterised
                    continue
            base = expr
        return base
    
    def getParameterisationParameters(self, name: str) -> list[str]:
        """ Gets the parameters that form the parametric expression of parameter `name`.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter was not parameterised.
        
        :return: A list of parameter names.
        :rtype: list
        """
        if name not in list(self.__parameterisation.keys()):
            raise ValueError("'%s' parameter is not parameterised." % name)
        
        return self.__parameterisation[name]["parameters"]
    
    def rmParameterisation(self, name: str) -> None:
        """ Unregisters the parameterisation of parameter `name`.
        
        :param name: The name of the parameter that is being parameterised.
        :type name: str
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: None
        :rtype: None
        """
        if name not in list(self.__collection.keys()):
            raise ValueError("'%s' parameter was not found." % name)
        
        if name not in list(self.__parameterisation.keys()):
            raise ValueError("'%s' parameter is not parameterised." % name)
        
        
        # If we want to remove the associated parameters, we'll also need to check they can actually be removed without breaking everything
        #for sname in self.__parameterisation[name]["parameters"]:
        #    del self.__collection[sname]
        names = self.__parameterisation[name]["parameters"]
        for pname in names:
            self.__parameterisation_graph.remove_edge(pname, name)
        del self.__parameterisation[name]
    
    def parameterisationParametersSet(self, name: str) -> bool:
        """ Checks if the parameters that parameterise `name` have been initialised.
        
        :param name: The name of the parameterised parameter.
        :type name: str
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: True if the parameters are initialised else False
        :rtype: bool
        """
        if name not in list(self.__parameterisation.keys()):
            raise ValueError("'%s' parameter was not found." % name)
        
        names = self.__parameterisation[name]['parameters']
        if None in self.getParameterValues(*names).values():
            return False
        return True
    
    def getParameterisationsInvolving(self, *names: str) -> list[str]:
        r""" Gets the list of parametric parameters that depend on the supplied parameter names. Returning an empty list if `name` is parametric parameter or doesn't exist in any parametric expressions.
        
        :param \*names: The names of the parameters.
        :type \*names: str
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: A list of parametric parameters. The list is empty if a parametric parameter is supplied as `name`.
        :rtype: list
        """
        pnames = []
        for name in list(names):
            if name not in list(self.__collection.keys()):
                raise ValueError("'%s' parameter was not found." % name)
            
            # FIXME: Parameterisations can depend on others
            if name in list(self.__parameterisation.keys()):
                return []
            
            for pname in self.getParametricParametersList():
                if name in self.getParameterisationParameters(pname):
                    if pname not in pnames:
                        pnames.append(pname)
        return pnames
    
    def getParameterisationSuccessors(self, name: str) -> dict[sy.Symbol, str]:
        return nx.dfs_successors(self.__parameterisation_graph, name)
    
    def drawParameterisationGraph(self, filename: str | None = None) -> gv.Source:
        # Get the pydot graph
        pd_graph = nx.nx_pydot.to_pydot(self.__parameterisation_graph)
        
        # Compile the graphviz source
        src = pd_graph.create(format='dot').decode('utf8')
        if filename is not None:
            with open(filename, "w") as fd:
                fd.write(src)
        return gv.Source(src)

    ###################################################################################################################
    #       Internal
    ###################################################################################################################
    
    def _get_pc_internal_data(self) -> _PCInternalData:
        return (
            self.__collection,
            self.__symbol_map,
            self.__parameterisation,
            self.__parameterisation_graph
        )
    
    def _set_pc_internal_data(self, data: _PCInternalData) -> None:
        self.__collection = data[0]
        self.__symbol_map = data[1]
        self.__parameterisation = data[2]
        self.__parameterisation_graph = data[3]
    
    # Use this with care, probably many scenarios where it would break things
    def _update_pc_internal_data(self, data: _PCInternalData) -> None:
        # Update the collection
        for param in data[0].keys():
            if param not in self.__collection.keys():
                self.__collection[param] = data[0][param]
        
        # Update the symbol map
        for param in data[1].keys():
            if param not in self.__symbol_map.keys():
                self.__symbol_map[param] = data[1][param]
        
        # Update the parameterisations
        #for param in data[2].keys():
        #    if param not in self.__parameterisation.keys():
        #        self.__parameterisation[param] = data[2][param]
        
        # Ok to just copy these for current use case
        self.__parameterisation = data[2]
        self.__parameterisation_graph = data[3]
    
    def updateParameterisations(self) -> None:
        """ Recomputes the values of all parameterised parameters from the currently set
        independent parameter values. Called automatically when parameter values are set;
        call it directly after registering a new parameterisation to make its value
        available immediately. """
        checked_nodes = set()
        checked_edges = set()
        G = self.__parameterisation_graph
        all_nodes = set(G.nodes)
        all_edges = set(G.edges)

        # Start with the nodes that have no in_degree, i.e. that are independent,
        # and get all the descendant paths.
        paths = {node: nx.descendants(G, node) for node in all_nodes if G.in_degree[node] == 0}

        # Get the parameters that need to have all values substituted
        dependent_nodes = all_nodes - set(paths)

        # For each path, update the substitutions using the independent parameters
        dependent_node_values = {k: None for k in dependent_nodes}
        for independent_node, path in paths.items():
            symbol = self.getSymbol(independent_node)
            value = self.getParameterValue(independent_node)
            if value is None:
                continue
            for dependent_node in path:
                if dependent_node_values[dependent_node] is None:
                    expression = self.getParametricExpression(dependent_node, expand=True)
                else:
                    expression = dependent_node_values[dependent_node]
                dependent_node_values[dependent_node] = expression.subs({symbol: value})

        # Update all the dependent parameter values we can
        for k, v in dependent_node_values.items():
            try:
                self.__collection[k].setValue(float(v))
            except:
                continue
