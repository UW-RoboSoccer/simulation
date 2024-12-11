import xml.etree.ElementTree as ET
import colorsys
import numpy as np

def list_filter(lambda_fn, iterable):
    return list(filter(lambda_fn, iterable))