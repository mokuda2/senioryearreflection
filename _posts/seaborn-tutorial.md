---
layout: post
title:  "Using the Iris Dataset with Seaborn"
author: Michael Okuda
description: Learn how to create basic visualizations with Seaborn.
image: /assets/images/seaborn-collaboration.jpg
---
## What is Seaborn?

Seaborn is a Python data visualization library based on Matplotlib. I personally like it over Matplotlib and Pandas visualizations because Seaborn provides more beautiful statistical graphics. Seaborn is designed to work with Pandas data frames and to make visualizations of complex data easier and more attractive. The following website is also a great resource for learning how to create visualizations in Seaborn: https://seaborn.pydata.org/tutorial.html.

## Getting Started with Seaborn

The first step is to install Seaborn. You can do this by running the following command in your terminal or command prompt:

**pip-install-seaborn

Next, you will need to import Seaborn into your Python environment. You can do this by using the following code:

**import-seaborn-as-sns

## Loading data into Seaborn

The following code will allow you access to the iris dataset, which is a data frame with 150 measurements of iris petal widths and lengths of three different species.

**iris

## Plotting with Different Visualizations

In this tutorial, we will go over the basics of how to create some common types of plots.  I will go over basic arguments, but the entire series of arguments can be found in the documentation for each type of graph.

## The Scatter Plot

The scatter plot uses two quantitative variables, one on the x-axis and the other on the y-axis.  More information on Seaborn scatter plots can be found here: https://seaborn.pydata.org/generated/seaborn.scatterplot.html.

The code below plots petal length on the x-axis and petal width on the y-axis with the "x=" and "y=" arguments, respectively.  The "data" argument specifies what data frame is being used, which is the "iris" variable.

**scatterplot

In the code below, the "hue" argument differentiates between levels of a variable and is most useful if the variable is categorical.  In this case, "species" has three different levels, and the legend shows which species is associated with what color.  This graphic is more descriptive than the previous graph, where we can see that the setosa species has the smallest petal length and petal width, whereas the virginica species overall has the longest petal length and petal width.

**scatterplot-with-hue

## The Boxplot

Boxplots are useful visualizations to compare the spread or variance of the data.  The bottom of the "box" is the 25th percentile, the middle line is the median, and the top of the "box" is the 75th percentile.  The code below shows three boxplots, where the species of iris is on the x-axis with the "x=" argument and the sepal length is on the y-axis with the "y=" argument.  Again, the "data=" argument specifies the data frame to be used.  We can see that setosa has the smallest sepal length compared to the other species.  We can also observe that there is an outlier in the virginica boxplot.

**boxplot

## The Kernel Density (KDE) Plot

The kernel density estimate (KDE) shows the distributions of certain data.  The code below shows sepal width on the x-axis with the "x=" argument, and "density" is automatically on the y-axis.  The "hue" argument differentiates between levels of a variable and is most useful if the variable is categorical.  The "species" variable is used to differentiate the three species of iris by color.  Again, the "data=" argument specifies that the data frame to be used.  To interpret this graph, we see that setosa overall has the longest sepal width.  The peak of the graph shows where most of the sepal width values lie.  For example, about one-third of the sepal width values for the setosa species is around 3.5.  However, we see that the versicolor and virginica species have about 40 percent on their sepal width values around 3.0.

**kdeplot

## The Pairplot

