# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:18:06 2016

@author: aa
"""
import matplotlib.pyplot as plt

def drawLine(p1,p2,**args):
   x=[p1[0],p2[0]]
   y=[p1[1],p2[1]]
   color=args.pop('color',None)
   linewidth=args.pop('linewidth',None)

   if color and linewidth:
       plt.plot(x,y,color,linewidth=linewidth)
   else:
       plt.plot(x,y)
