# project/preprocessing_utils.py

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler 
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

    
    default_options = {
        'handle_unknown': 'ignore', 
        'sparse_output': False      
    }
    
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

    
    default_options = {
        'handle_unknown': 'use_encoded_value', 
        'unknown_value': -1                    
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


