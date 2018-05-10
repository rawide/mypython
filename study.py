#! /usr/bin/python
# -*- coding: utf-8 -*-

import functools
int16 = functools.partial(int, base = 16)
print(int16('abddf'))
