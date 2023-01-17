# -*- coding: utf-8 -*-
""" Mainly opening files + some pre-processing """

def get_source_code(html_path):
    """ Return graph visualisation HTML """
    with open(html_path, 'r', encoding='utf-8') as html_file:
        source_code = html_file.read()
    return source_code
