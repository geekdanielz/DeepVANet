"""
Util functions
"""


def out_put(string, verbose):
    '''
    Help function for verbose,
    output the string to destination path

    Parameters
    ----------
    string  :str,  the string to output
    verbose :str, the path to store the output
    '''
    with open(f"{verbose}.txt", "a") as f:
        f.write(string + "\n")
