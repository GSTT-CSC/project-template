# project/preprocessing_utils.py

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler # Good to include for numerical data
#from category_encoders import TargetEncoder


def get_onehot_encoder(options=None):
    """
    Returns a configured OneHotEncoder instance.

    Args:
        options (dict, optional): Dictionary of keyword arguments to pass
                                  to the OneHotEncoder constructor.
                                  Common options:
                                  - 'handle_unknown': 'ignore' or 'error' (default 'error')
                                  - 'sparse_output': True or False (default True)
                                  - 'drop': 'first', 'if_binary', or a list of categories to drop.

    Returns:
        sklearn.preprocessing.OneHotEncoder: Configured OneHotEncoder instance.
    """
    if options is None:
        options = {}

    # Default options for robustness in a pipeline
    default_options = {
        'handle_unknown': 'ignore',  # Important for handling new categories in test set
        'sparse_output': False       # Often preferred for simpler integration with XGBoost/Pandas
    }
    # Merge default options with user-provided options,
    # user options will override defaults
    encoder_options = {**default_options, **options}

    return OneHotEncoder(**encoder_options)


def get_ordinal_encoder(options=None):
    """
    Returns a configured OrdinalEncoder instance.

    Args:
        options (dict, optional): Dictionary of keyword arguments to pass
                                  to the OrdinalEncoder constructor.
                                  Common options:
                                  - 'categories': 'auto' or a list of lists for explicit order.
                                  - 'handle_unknown': 'error' or 'use_encoded_value' (default 'error')
                                  - 'unknown_value': value to use for unknown categories if handle_unknown='use_encoded_value'.

    Returns:
        sklearn.preprocessing.OrdinalEncoder: Configured OrdinalEncoder instance.
    """
    if options is None:
        options = {}

    # Default options for robustness
    default_options = {
        'handle_unknown': 'use_encoded_value', # Recommended for unknown categories in test set
        'unknown_value': -1                    # A common choice for unknown value
    }
    encoder_options = {**default_options, **options}

    return OrdinalEncoder(**encoder_options)


def get_standard_scaler(options=None):
    """
    Returns a configured StandardScaler instance.

    Args:
        options (dict, optional): Dictionary of keyword arguments to pass
                                  to the StandardScaler constructor.

    Returns:
        sklearn.preprocessing.StandardScaler: Configured StandardScaler instance.
    """
    if options is None:
        options = {}
    return StandardScaler(**options)


# You can add more encoder-related utility functions here:
# def get_target_encoder(options=None):
#    if options is None:
#        options = {}
#    return TargetEncoder(**options)