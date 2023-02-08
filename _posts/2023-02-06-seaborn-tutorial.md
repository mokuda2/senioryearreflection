---
layout: post
title:  "Creating Visually Appealing Visualizations with Seaborn and the Iris Dataset"
author: Michael Okuda
description: Learn how to create basic visualizations with Seaborn.
image: /assets/images/seaborn-collaboration.jpg
---
## What is Seaborn?

Seaborn is a Python data visualization library based on Matplotlib. I personally like it over Matplotlib and Pandas visualizations because Seaborn provides more beautiful statistical graphics. Seaborn is designed to work with Pandas data frames and to make visualizations of complex data easier and more attractive. The following website is also a great resource for learning how to create visualizations in Seaborn: https://seaborn.pydata.org/api.html.

## Getting Started with Seaborn

The first step is to install Seaborn. You can do this by running the following command in your terminal or command prompt:

_**pip install seaborn**_

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/pip-install-seaborn.png)

Next, you will need to import Seaborn into your Python environment. You can do this by using the code below.  The alias "sns" is a standard abbreviation for "Seaborn."

_**import seaborn as sns**_

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/import-seaborn2.png)

## Loading data into Seaborn

The following code will allow you access to the iris dataset, which is a data frame with 150 measurements of iris petal widths and lengths of three different species.

_**iris = sns.load_dataset("iris)**_\
_**iris**_

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/iris-dataframe2.png)

## Plotting with Different Visualizations

In this tutorial, we will go over the basics of how to create some common types of plots.  I will go over basic arguments, but the entire series of arguments can be found in the documentation for each type of graph.

## The Scatter Plot

The scatter plot graphs two quantitative variables against each other, one on the x-axis and the other on the y-axis.  More information on Seaborn scatter plots can be found here: https://seaborn.pydata.org/generated/seaborn.scatterplot.html.

From the code below, the following arguments are used:

_**sns.scatterplot(x="petal_length", y="petal_width", data=iris)**_

* "x=": the variable plotted on the x-axis.  Petal length is plotted on the x-axis.
* "y=": the variable plotted on the y-axis.  Petal width is plotted on the y-axis.
* "data=": specifies which data frame to use.  In this case, it is the "iris" variable.

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/scatterplot-without-hue.png)

* "hue=": differentiates between levels of a variable and is most useful if the variable is categorical.  In the code below, "species" has three different levels, and the legend shows which species is associated with what color.

Analysis: This graphic is more descriptive than the previous graph, where we can see that the setosa species has the smallest petal length and petal width, whereas the virginica species overall has the longest petal length and petal width.

_**sns.scatterplot(x="petal_length", y="petal_width", hue="species", data=iris)**_

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/scatterplot-with-hue.png)

## The Boxplot

Boxplots are useful visualizations to compare the spread or variance of the data.  The bottom of the "box" is the 25th percentile, the middle line is the median, and the top of the "box" is the 75th percentile.  More information on Seaborn boxplots can be found here: https://seaborn.pydata.org/generated/seaborn.boxplot.html.

From the code below, the following arguments are used:

_**sns.boxplot(x="species", y="sepal_length", data=iris)**_

* "x=": the variable plotted on the x-axis.  "Species" is plotted on the x-axis.
* "y=": the variable plotted on the y-axis.  Sepal length is plotted on the y-axis.
* "data=": specifies which data frame to use.  In this case, it is the "iris" variable.

Analysis: We can see that setosa has the smallest sepal length compared to the other species.  We can also observe that there is an outlier in the virginica boxplot.

## The Kernel Density (KDE) Plot

The kernel density estimate (KDE) shows the distributions of certain data.  More information on Seaborn KDEs can be found here: https://seaborn.pydata.org/generated/seaborn.kdeplot.html.

From the code below, the following arguments are used:

_**sns.kdeplot(x='sepal_width', hue='species', data=iris)**_

* "x=": the variable plotted on the x-axis.  Sepal width is plotted on the x-axis.
* "hue=": differentiates between levels of a variable and is most useful if the variable is categorical.  In the code below, "species" has three different levels, and the legend shows which species is associated with what color.
* "data=": specifies which data frame to use.  In this case, it is the "iris" variable.

Analysis: To interpret this graph, we see that setosa overall has the longest sepal width.  The peak of the graph shows where most of the sepal width values lie.  Notice that the y-axis automatically plots the density of the KDE.  For example, about one-third of the sepal width values for the setosa species is around 3.5.  However, we see that the versicolor and virginica species have about 40 percent on their sepal width values around 3.0.

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/kdeplot.png)

## The Pairplot

Pairplots graph every combination pair of quantitative variables. More information on Seaborn pair plots can be found here: https://seaborn.pydata.org/generated/seaborn.pairplot.html.

From the code below, the following arguments are used:

_**sns.pairplot(data=iris, hue="species")**_

* "data=": specifies which data frame to use.  In this case, it is the "iris" variable.
* "hue=": differentiates between levels of a variable by color and is most useful if the variable is categorical.  The "species" column is used to color the different species of iris.

Analysis: We can see that scatter plots and KDEs are graphed in the pairplot, including the scatter plot and KDE plot that were graphed in the above examples.  This type of graph is useful for a comprehensive overview of the quantitative variables in the dataset.

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/pairplot.png)

## A Call to Action

Now that we've graphed a few visualizations with Seaborn, we can apply what we've learned and more with other datasets.  The "titanic" dataset is also well-known and has data about the passengers of the Titanic.  What visualizations can you create?  What trends do you find from exploring the data?  The code below loads and shows the "titanic" data frame.

**_titanic = sns.load_dataset("titanic")_**\
**_titanic_**