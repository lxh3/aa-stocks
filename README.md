**Companion Code for:**
****************************************************************
****************************************************************
# Canonical sectors and evolution of firms 
# in the US stock markets
****************************************************************
****************************************************************
**Citation:**

> Lorien X. Hayden, Ricky Chachra, Alexander A. Alemi,
> Paul H. Ginsparg & James P. Sethna (2018)
> Canonical sectors and evolution of firms in the US stock markets,
> Quantitative Finance, DOI: 10.1080/14697688.2018.1444278

> https://arxiv.org/abs/1503.06205

**Last updated 09/17/2018 by Lorien Hayden**

****************************************************************

## Use:
****

```console
foo@bar:~$ python main.py -v
```
The flag '-v' is an option to print the progress of code as it runs. May be omitted.

main.py
1) Performs all of the calculations necessary to produce the figures and table data in the paper (exception: see 'To Do' at end of README.txt).
2) Saves decompositions and other calculations in 'aa-stocks/saves/'
3) Produces figures in png format, table data in txt/LaTex format, and sankey data in json format. Saves output to folder 'aa-stocks/figures/'



## External Code Sources:
**********************

**Archetypal Analysis:**
archetypal_analysis.py is a python implementation of the MATLAB code provided accompanying *Morup, M. and Hansen, L.K., Archetypal analysis for machine learning and data mining. Neurocomputing,2012,80,54-63*

**Collect Stock Data:**
stock data was originally collected using ystockquote.py by Corey Goldberg available at https://github.com/cgoldberg/ystockquote/blob/master/ystockquote.py

**Produce Sankey Diagram:**
The sankey diagram is rendered using the Sankey extension provided by Mike Bostock available at http://bost.ocks.org/mike/sankey/


## To Do:
**********************
* Update plotting functions to produce figures in latest formatting
* Add linear regressions which compare the explanatory power of the three sector archetypal analysis decomposition with the market cap decomposition of Fama and French
                                                                      
