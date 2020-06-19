"""
DRAW

This module defines a method to draw initial and final messages of the code.

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0.1
"""

def walle(text):
    """This method prints a text together with a wall-e image so that the text
    is a bit more fun.

    Parameters:
    -----------
    text: str
        Text to show

    Returns:
    str
        An ASCII image of wall-e with the input text
    """
    print("""
            _  _
          /u@)(@\        """+text+"""
     nn      Y
     ' Y_____H____
      \ V  |-['']-|___,,
       \ \ |T'T' T|___nn
       |   || |  ||   UU
       /``\WALL.E `\\
      / /A \   / A  \\
      L____J   L____J
    """)
