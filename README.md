--// FIFA 22 Player Data Analysis \\--

This project performs exploratory data analysis and visualization on the FIFA 22 player dataset.
The dataset used is the FIFA 22 player dataset from Kaggle.


//-- Features --\\

- Loads player data from a CSV file

- Computes and exports preliminary numeric and categorical statistics

- Visualizes player data using plots:
  - Boxplot
  - Violinplot
  - Histogram
  - Conditional histogram
  - Error bar plot
  - Heatmap  
  - Estimated regression fit plot


//-- Requirements --\\

To install the required Python packages, use:

pip install -r requirements.txt


--// Usage \\--
Make sure the dataset file players_22.csv is in the same directory as the script. Then, run:


python data_analysis_main.py
	

The script will:

   - Load the dataset.

   - Generate and save statistics to numeric_stats.csv and categorical_stats.csv.

   - Display various visualizations to help explore the dataset.
