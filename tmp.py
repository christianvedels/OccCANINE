# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:49:29 2024

@author: chris
"""

if __name__ == '__main__':
    from histocc.class_activation.diagnostics import StringGradient
    
    x = StringGradient(n=3, verbose = True)
    
    
    x.visualize_text_gradients("agent for sewing machine company", 631)
    x.visualize_gradients("agent for sewing machine company", 631)
    
    # x = ClassActivation()
    
    # for i in range(3, 1919):
    #     res_i, prob = x.max_class_change(i, verbose = True)
    #     print(f"---> {i}: {res_i} -- {prob:.4f}")
    
    y=2
    
    
    
    
    