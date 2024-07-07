# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:49:29 2024

@author: chris
"""

if __name__ == '__main__':
    from histocc.diagnostics.class_activation import StringGradient
        
    text = "agent for sewing machine company"

    x = StringGradient(n=2, verbose = True)

    x.visualize_text_gradients(text, 631, what="logits")
    x.visualize_gradients(text, 631, what = "logits")