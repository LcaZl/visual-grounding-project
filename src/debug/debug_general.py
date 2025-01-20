import matplotlib.pyplot as plt
from tabulate import tabulate
import torch
import seaborn as sns
import pandas as pd


def plot_matrix(matrix: torch.Tensor, max_x: int = 100, max_y: int = 100, xlabel: str = 'X', ylabel: str = 'Y', title: str = 'No title', figsize: tuple = (12, 12)) -> None:
    """
    Plot a heatmap of a matrix with optional axis labels and title.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to visualize.
    max_x : int, optional
        Maximum number of columns to display (default is 100).
    max_y : int, optional
        Maximum number of rows to display (default is 100).
    xlabel : str, optional
        Label for the x-axis (default is 'X').
    ylabel : str, optional
        Label for the y-axis (default is 'Y').
    title : str, optional
        Title of the plot (default is 'No title').
    figsize : tuple, optional
        Size of the figure (default is (12, 12)).

    """
    resized_matrix = matrix[:max_x, :max_y] if max_x and max_y else matrix

    plt.figure(figsize=figsize)
    sns.heatmap(resized_matrix, annot=True, cmap='coolwarm', linecolor='white', linewidths=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    
def print_list(lis: list, title: str = "") -> None:
    """
    Print each element of a list with a title.

    Parameters
    ----------
    lis : list
        List of elements to print.
    title : str, optional
        Title for the printed list (default is "").

    """
    print(title)
    if len(lis) == 0:
        print(" -- Empty list.")
    else:
        for i, el in enumerate(lis, 1):
            print(f" - {i} - {el}")

   
def print_map(config_map: dict, title: str = None) -> None:
    """
    Print the key-value pairs of a dictionary with an optional title.

    Parameters
    ----------
    config_map : dict
        The dictionary to print.
    title : str, optional
        Title for the printed map (default is None).

    """
    if title:
        print(title)
        
    for i, (key, value) in enumerate(config_map.items(), 1):
        print(f"{i} - {key} : {value}")

                 
def print_dataframe(data: pd.DataFrame, title: str = None, limit: int = 30, sort_by: str = None, ascending: bool = True, show_index: bool = True) -> None:
    """
    Print a DataFrame with an optional title and sorting.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to print.
    title : str, optional
        Title for the printed DataFrame (default is None).
    limit : int, optional
        Maximum number of rows to display (default is 30).
    sort_by : str, optional
        Column to sort by (default is None).
    ascending : bool, optional
        Whether to sort in ascending order (default is True).
    show_index : bool, optional
        Whether to display the index (default is True).

    """
    if title:
        print(title)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    # Handle Series as DataFrame
    if isinstance(data, pd.Series):
        column_name = data.name if data.name else 'Value'
        data = data.to_frame(name=column_name)

    if sort_by:
        data = data.sort_values(by=sort_by, ascending=ascending)
    
    print("\n\n", tabulate(data[:limit], headers='keys', tablefmt='simple_grid', showindex=show_index))
    print('\n')

def print_dict(d: dict, title: str = '', avoid_keys: list = []) -> None:
    """
    Print a dictionary's key-value pairs with an optional title, excluding specified keys.

    Parameters
    ----------
    d : dict
        The dictionary to print.
    title : str, optional
        Title for the printed dictionary (default is '').
    avoid_keys : list, optional
        List of keys to exclude from printing (default is []).


    """
    if title:
        print(title)
    
    for key, value in d.items():
        if key not in avoid_keys:
            if isinstance(value, torch.Tensor):
                print(f" - {key} (tensor) - Shape:{value.shape} - Type:{value.dtype} - Device:{value.device}")
            else:
                print(f" - {key}: {value}")

            
            