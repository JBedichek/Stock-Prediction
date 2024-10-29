* Stock Prediction *

Rundown of the project components:

models.py - Definition of the PyTorch models.

stock.py - Data Scraping

inference.py - Running trading simulations using the models

training.py - Dataset creation, preprocessing, training. 

utils - Misc functions for use in training, inference, or both.



Architecture:

The model takes 600 market days (3 years) of stock information for an individual company, and makes
a prediction over a discrete probability distribution over the next 4 days of the relative change in price of 
that stock.

In addition to the individual stock, it also takes 3 years of market movement. 
There are 5 pieces of stock information which are defined every market day for a stock

open, high, low, close, volume



This code was not made with the intention of being easy or intuitive to read by others, so it may appear somewhat disorganized.
