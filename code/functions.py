# This is where your figures will be saved when calling eval_model().
fig_path = '/Users/ekinderdiyok/Documents/Thesis/Figures/' 

# Input where your files are stored
data_path = '/Users/ekinderdiyok/Documents/Thesis/Data/'

# Needed for tagging topics and determining semantic category 
my_openai_api_key = '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'

import pandas as pd
import random
import datetime
import matplotlib
import math
import sys
import statistics
import openai # for identfying tweet topics
import numpy as np
import networkx as nx # for creating network graphs
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
from matplotlib.ticker import FuncFormatter
import ndlib
import ndlib.models.ModelConfig as mc # for epidemiological models
import ndlib.models.epidemics as ep # for epidemiological models
import scipy.optimize as optimize # For ordinary least squares
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
import scipy
from scipy.stats import f_oneway
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def plot_rt_network(data, tweet_id, fn, pct_exc = 0.97, only_rt = False):

    # Create graph
    G = create_G(data, tweet_id, fn)

    # Select original poster
    original_poster = data.loc[tweet_id].user_id
    
    # Exclude 90% of nodes with zero outdegree, i.e., nodes who has not retweeted.
    nodes_to_exc = select_nodes_to_exc(G, pct_exc)

    # Remove reflexive edges, i.e., self-pointing, circular edges. Data have a problem that is some users have multiple user_ids.
    G = remove_reflexive_edges(G)
    
    # Remove nodes from graph G
    G.remove_nodes_from(nodes_to_exc)

    # Create a list of nodes that has zero outdegree - after removing 97% of them.
    followers = [node for node in G.nodes() if G.out_degree(node) == 0]

    # Draw graph G
    draw_G(G, original_poster = original_poster, followers = followers)

    # Add title
    plt.title("The aggregate follower network of users who participated in a cascade")

    # Add legend manually to be able to mark different nodes with different colors
    leg_op = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Source node')
    leg_rt = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#505050', markersize=10, label='Retweeter (center)', alpha=0.4)
    leg_edge = plt.Line2D([0], [0], color='#e2e2e2', label='Followed by')
    if only_rt is True:
        leg_fw = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#000000', markersize=10, label='Follower (periphery)')
        plt.legend(handles=[leg_op,leg_rt, leg_fw, leg_edge], loc = 'upper left')
    plt.legend(handles=[leg_op,leg_rt, leg_edge], loc = 'upper left')
    
    return None    

def ttest_cats(cat1, cat2, alpha):
    t_statistic, p_value = ttest_ind(cat1, cat2)
    
    if p_value < alpha:
        print('Reject the null hypothesis: There is a significant difference between the two categories.')
    else:
        print('Fail to reject the null hypothesis: There is no significant difference between the two categories.')
    
    # Calculate degrees of freedom 	
    degrees_of_freedom = (len(cat1) + len(cat2) - 2) // 2

    print(f'T-Statistic: {t_statistic}')
    print(f'Degrees of Freedom: {degrees_of_freedom}')
    print(f'P-Value: {p_value}')
    return None

def set_plot_style():
    """
    Set the plot style for matplotlib and seaborn.

    This function updates the default parameters for matplotlib to control
    the font sizes for text, axis labels, titles, and axis ticks. It also
    sets the plot style to white background using seaborn and matches the
    font to LaTeX font.

    Parameters:
    None

    Returns:
    None
    """
    plt.rcParams.update({
        'font.size': 12,          # Default font size for text
        'axes.labelsize': 12,     # Font size for x and y axis labels
        'axes.titlesize': 12,     # Font size for titles
        'xtick.labelsize': 12,    # Font size for x-axis ticks
        'ytick.labelsize': 12     # Font size for y-axis ticks
    })

    # Set the plot style to white background
    sns.set_style('white')

    # Match LaTeX font
    font = {'family' : 'serif',
             'size'   : 12,
             'serif':  'cmr10'
             }
    matplotlib.rc('font', **font)
    return None

def eval_model(input_function, param, data, n_iter, fn, k = None):
    """
    Evaluate a simulation model by comparing simulated and observed retweet counts using various metrics.

    Parameters:
    - input_function (callable): A simulation function to evaluate.
    - param (float or tuple): Parameters required for the simulation function.
    - data (pandas.DataFrame): A DataFrame containing observed retweet counts, indexed by tweet_id.
    - n_iter (int): The number of iterations for the simulation.
    - k (int, optional): A parameter specific to certain simulation functions (default is None).

    Returns:
    - mad (float): Model mean absolute deviation, a measure of the average absolute difference between
                   simulated and observed retweet counts.
    - correlation_coefficient (float): Pearson's correlation coefficient between simulated and observed retweet counts.
    - sims (list): List of simulated retweet counts for each tweet.
    - obss (list): List of observed retweet counts for each tweet.

    Notes:
    - The function iterates over each tweet in the provided data, simulates retweet counts using the specified
      simulation function, and compares the results with the observed retweet counts.
    - The mad is calculated as the sum of absolute differences divided by the number of tweets.
    - Pearson's correlation coefficient measures the linear correlation between simulated and observed retweet counts.
    - The function also generates a scatter plot comparing observed and simulated retweet counts, along with an ideal fit line.

    Example:
    ```python
    mse, corr_coef, _, _ = eval_model(simulate_IC, 0.0055, cascades, n_iter = 5)
    ```

    """
    
    sims = []
    obss = []
    ss = 0
    tweet_ids = data.index.values
    for tweet_id in tweet_ids:
        if input_function.__name__ == 'sim_decaying_IC':
            G = create_G(data, tweet_id, fn)
            sim, obs, _ = input_function(data, G, param, tweet_id, n_iter, k)
        elif input_function.__name__ in ['simulate_SI', 'simulate_IC']:
            G = create_G(data, tweet_id, fn)
            sim, obs = input_function(data, G, param, tweet_id, n_iter)
        elif input_function.__name__ == 'simulate_base_SI':
            sim, obs = input_function(data, param, tweet_id, n_iter)
       
        sims.append(sim)
        obss.append(obs)

    tad = np.sum(abs(np.array(sims) - np.array(obss))) 
    mad = tad/len(tweet_ids)
    print(f'Model mean absolute deviation is: {mad}')
    
    # Calculate Pearson's correlation coefficient
    correlation_coefficient, p_value = pearsonr(sims, obss)
    print("Pearson's correlation coefficient:", correlation_coefficient)
    print("P-value:", p_value)
    
    # Create a scatter plot
    plt.scatter(obss, sims, marker='o', facecolors='None', edgecolors='gray', label = 'Cascade')

    # Add labels and a title
    plt.xlabel('Observed retweet count',labelpad=20)
    plt.ylabel('Simulated retweet count',labelpad=20)
    plt.title('Observed vs simulated retweet count',pad=20)
    plt.xscale('log')
    plt.yscale('log')

    # Plot the y = x line
    plt.plot([min(obss), max(obss)], [min(obss), max(obss)], linestyle='--', color='red', label='Ideal fit')
    
    # Display a legend
    plt.legend()

    sns.despine()
    
    #plt.savefig(fig_path + input_function.__name__ + '_plot.svg', format='svg', bbox_inches='tight')
    #plt.savefig(fig_path + input_function.__name__ + '_plot.png', format='png', bbox_inches='tight', dpi = 300)
    
    # Show the plot
    plt.show()

    return mad, correlation_coefficient, sims, obss





def create_G(data, tweet_id, fn):
    """
    Creates a directed graph representing the propagation cascade of a given tweet.

    Parameters:
    - data (pd.DataFrame): A DataFrame containing information about tweets, followers, and retweeters.
    - tweet_id (str): The unique identifier of the tweet for which the propagation graph is to be created.

    Returns:
    - G (nx.DiGraph): A directed graph (NetworkX DiGraph) representing the propagation cascade.

    Algorithm:
    1. Create an empty directed graph G using NetworkX library.
    2. Retrieve information about the original poster and related data.
    3. Iterate through the followers of the original poster and add edges to G.
    4. Extract followers of retweeters from the dataset.
    5. Iterate through the retweeters, adding edges from retweeter to their followers.
    
    Example Usage:
    ```python
    tweet_id = pick_random_tweet_id(cascades)
    G = create_G(cascades, tweet_id, fn)
    inspect_G(G)
    ```

    Note:
    - The function assumes that the input DataFrame 'data' has columns 'user_id', 'followers', and 'retweeters'.
    - 'pick_random_tweet_id' is a function that selects a random tweet_id from a given dataset.
    - 'inspect_G' is a function for visualizing or analyzing the created graph.
    """
    
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Define nodes
    original_poster = data.loc[tweet_id]  # pd.Series
    followers = original_poster.followers  # list
    
    retweeters = original_poster.retweeters  # list
    followers_of_retweeters = fn[fn.user_id.isin(retweeters)]  # pd.DataFrame
    
    # Iterate through the followers of the original_poster and add edges.
    for follower in followers:
        source = original_poster.user_id  # user_id of original_poster
        target = follower
        G.add_edge(source, target)
    
    # Iterate over retweeters of the original_poster
    for _, retweeter in followers_of_retweeters.iterrows():
        source = retweeter['user_id']
        targets = retweeter['followers']
        
        # Iterate over followers of a single retweeter and add edges.
        for target in targets:
            G.add_edge(source, target)
    
    return G





def pick_random_tweet_id(data):
    """
    Randomly select a tweet_id from the provided dataset.

    Parameters:
    - data (pandas.DataFrame): The dataset containing tweet information.

    Returns:
    - tweet_id (hashable): Randomly chosen tweet_id.

    The function samples a random tweet_id from the given dataset and prints the chosen tweet_id.
    """
    
    # Pick a random tweet id 
    tweet_id = data.sample().index.item() #str
    
    # Print it
    print(f"Randomly chosen tweet_id is: '{tweet_id}'")
    
    return tweet_id





def scale_timestamps(timestamps):
    """
    Scale timestamps by making them start at zero and converting to hours.

    Parameters:
    - timestamps (list of int/float): List of timestamps to be scaled.

    Returns:
    - scaled_timestamps (list of float): Scaled timestamps in hours.

    The function makes timestamps start at zero by subtracting the minimum value,
    then converts the adjusted timestamps to hours. The resulting scaled timestamps
    are returned as a list of floats.
    """
    
    # Make timestamps start at zero
    timestamps = [timestamp - min(timestamps) for timestamp in timestamps]

    # Convert timestamps to hours
    timestamps = [timestamp / 3600 for timestamp in timestamps]

    return timestamps





def create_bootstrap_samples(data, n_bootstrap_samples, n_rows, random_seed):
    """
    Create bootstrap samples from the given dataset.

    Parameters:
    - data (pandas.DataFrame): The dataset to create bootstrap samples from.
    - n_bootstrap_samples (int): Number of bootstrap samples to generate.
    - n_rows (int): Number of rows to include in each bootstrap sample.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - bootstrap_samples (list of pandas.DataFrame): List of bootstrap samples.

    The function generates bootstrap samples by randomly sampling rows from the input dataset
    with replacement. It prints the number of duplicate rows in the first bootstrap sample and
    the total number of bootstrap samples created. The resulting bootstrap samples are returned
    as a list of pandas DataFrames.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Create a bootstrap sample
    bootstrap_samples = [data.sample(n=n_rows, replace=True) for _ in range(n_bootstrap_samples)]

    # Print information about duplicates and number of samples
    print(f'n_rows that are duplicates: {bootstrap_samples[0].index.duplicated().sum()}')
    print(f'{len(bootstrap_samples)} bootstrap samples were created')

    return bootstrap_samples





def inspect_G(cascades, G, tweet_id):
    
    original_poster = cascades.loc[tweet_id, 'user_id']
    print(f'The number of nodes in graph G are {G.number_of_nodes()}')
    print(f'The number of edges in graph G are {G.number_of_edges()}')
    print(f"The number of retweets is {cascades.loc[tweet_id, 'n_rts']}")
    print(f'Out-degree of the original poster: {G.out_degree(original_poster)}')
    print(f'In-degree of the original poster: {G.in_degree(original_poster)}')
    print(f"Number of network followers of the original poster: {cascades.loc[tweet_id, 'n_network_followers']}")

    # Extract out-degrees
    out_degrees = list(G.out_degree)
    
    # Extract the second element of each tuple
    second_elements = [tup[1] for tup in out_degrees]

    """ # Draw a histogram
    plt.figure(figsize=(16, 9))  # Set a larger figure size
    plt.hist(second_elements, bins=50, edgecolor='black')
    plt.xlabel('Count of followers')
    plt.ylabel('Frequency')
    plt.title('Number of followers of followers')
    plt.yscale('log')
    plt.show()"""





def ident_topic_clust(tweets, api_key, model="gpt-3.5-turbo", max_tokens=300):
    """
    Identify major topic clusters in a collection of tweets using OpenAI GPT-3.5-turbo.

    Parameters:
    - tweets (str): A collection of tweets to analyze for topic clusters.
    - api_key (str, optional): Your OpenAI API key. If not provided, it uses the default value set during API initialization.
    - model (str, optional): OpenAI language model to use. Default is "gpt-3.5-turbo".
    - max_tokens (int, optional): Maximum number of tokens in the generated response. Default is 300.

    Returns:
    - clusters (str): Identified major topic clusters generated by the language model.

    The function sends a request to the OpenAI GPT-3.5-turbo model to identify major topic clusters
    in the provided collection of tweets. It returns the generated clusters as a string.

    Usage:
    1. Provide a collection of tweets as a string to the 'tweets' parameter.
    2. Optionally, provide your OpenAI API key to the 'api_key' parameter. If not provided, the default value is used.
    3. Optionally, specify the OpenAI language model to use with the 'model' parameter. Default is "gpt-3.5-turbo".
    4. Optionally, set the maximum number of tokens in the generated response with the 'max_tokens' parameter. Default is 300.

    Example:
    ```python
    chunks = np.array_split(cascades['text'], 3)
    
    clusters1 = ident_topic_clust(tweets=chunks[0].str.cat(sep=" \n "))
    clusters2 = ident_topic_clust(tweets=chunks[1].str.cat(sep=" \n "))
    clusters3 = ident_topic_clust(tweets=chunks[2].str.cat(sep=" \n "))
    
    print(clusters1)
    print(clusters2)
    print(clusters3)
    ```
    """
    # Set the first part of your prompt
    fixed = """The following is a large collection of tweets about the Nepal 2015 earthquake. Identify major  topic clusters:"""

    # Combine the fixed part of the prompt with the tweet
    prompt = fixed + tweets

    # Set your OpenAI API key
    if api_key is not None:
        openai.api_key = api_key

    # Make a request to the API
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    # Extract and return the generated tags
    clusters = response.choices[0]['message']['content']
    return clusters





def ident_tweet_topic(tweet, api_key, model="gpt-3.5-turbo-1106", max_tokens=20):
    
    # Set the first part of your prompt
    fixed = """Only write the name of the topic which fits the following tweet best. Choose one of the following possible answers:
    1. Rescue operations and assistance
    2. Impact, aftermath, death toll
    3. Aftershocks and tremors
    4. International response and international support
    5. Technology, communication, and social media
    6. Controversy, criticism, social and political commentary
    7. Relief efforts, resources, donations, and infrastructure
    8. Faith and prayers
    9. Personal experience, stories, and missing people
    10. The tweet does not fit any of the given topics.
    Following text is the tweet: """
    
    # Combine the fixed part of the prompt with the tweet
    prompt = fixed + tweet
    
    # Set your OpenAI API key
    openai.api_key = api_key

    # Make a request to the API
    response = openai.ChatCompletion.create(
        model=model,
        temperature = 0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    # Extract and return the generated tags
    topic = response.choices[0]['message']['content']
    return topic





def ident_fact_vs_emot(tweet, api_key, model="gpt-3.5-turbo-1106", max_tokens=300):

    # Set your OpenAI API key
    openai.api_key = api_key

    # Set the fixed part of your prompt
    fixed = """Write 0 if the following tweet is a factual tweet. Write 1 if the following tweet is a personal or emotional tweet. Only answer by writing 0 or 1. Here is the tweet: """

    # Combine the fixed part of the prompt with the tweet
    prompt = fixed + tweet

    # Make a request to the API
    response = openai.ChatCompletion.create(
        model=model,
        temperature = 0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    # Extract and return the generated tags
    fact_vs_emot = response.choices[0]['message']['content']
    return fact_vs_emot

def plot_param_dicts(param_dicts):
    
    # Create figure
    plt.figure(figsize=(16/2, 9/2))

    # Get the current Axes instance
    ax = plt.gca()

    # Set the font size of the x-axis ticks
    ax.tick_params(axis='x', labelsize=16)  

    # Set the font size of the y-axis ticks
    ax.tick_params(axis='y', labelsize=16)  
    labels = list(param_dicts.keys())
    labels.reverse()
    values = [t[0] for t in param_dicts.values()]
    sizes = [t[1] for t in param_dicts.values()]
    values.reverse()
    
    # Creating the point plot
    plt.scatter(values, labels, color='gray', marker='o', s = sizes)

    # Adding labels and title
    plt.xlabel('X-axis Label',labelpad=20)
    plt.ylabel('Y-axis Label')
    plt.title('Point Plot from Dictionary')

    # Annotating each marker with integers from 'sizes'
    for i, txt in enumerate(sizes):
        plt.annotate(txt, (values[i], labels[i]), fontsize=12, ha='left', va='center')

    # Display the plot
    plt.show()
    
    return None

def split_dataframe_by_category(dataframe, category_column):
    """
    Splits a DataFrame into smaller DataFrames based on a categorical variable.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to be split.
        category_column (str): The name of the column used for splitting.

    Returns:
        dict: A dictionary where keys are unique values from the specified column,
              and values are the corresponding smaller DataFrames.
    """
    grouped = dataframe.groupby(category_column)
    groups = {}
    for category, group in grouped:
        groups[category] = group
    return groups


def preprocess_data():
    
    """
    Preprocesses data related to tweet cascades, including user series, time series, and followers network.

    Reads data from CSV files, performs cleaning and merging operations,
    and creates additional columns for analysis. The resulting DataFrame is split
    into groups based on the 'topic' column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]:
            - cascades (pd.DataFrame): Processed data containing tweet cascade information.
            - fn (pd.DataFrame): Processed followers network data.
            - groups (dict): Dictionary of DataFrames, where keys are unique 'topic' values,
                             and values are corresponding DataFrames.
    """
    
    # Read the user series data from a CSV file
    us = pd.read_csv(data_path + 'userseries.txt', header=None)

    # Split the first column into 'tweet_id' and 'retweeters' columns
    us[['tweet_id', 'retweeters']] = us.iloc[:, 0].str.split(' ', 1, expand=True)

    # Drop the old column and set 'tweet_id' as the index
    us = us[['tweet_id', 'retweeters']]
    us.set_index('tweet_id', inplace=True)

    # Read the time series data from a CSV file
    ts = pd.read_csv(data_path + 'timeseries.txt', header=None)

    # Split the first column into 'tweet_id' and 'timestamps' columns
    ts[['tweet_id', 'timestamps']] = ts.iloc[:, 0].str.split(' ', 1, expand=True)

    # Drop the old column and set 'tweet_id' as the index
    ts = ts[['tweet_id', 'timestamps']]
    ts.set_index('tweet_id', inplace=True)

    # Merge the user series and time series data to get 'retweeters' for each tweet
    cascades = us.merge(ts, left_index=True, right_index=True, how='inner')

    # Clean the 'retweeters' column by replacing multiple whitespaces with a single space
    cascades['retweeters'] = cascades['retweeters'].str.replace(r'\s+', ' ', regex=True)

    # Remove leading and trailing whitespaces from the 'retweeters' column
    cascades['retweeters'] = cascades['retweeters'].str.strip()

    # Calculate the number of retweets (number of words in 'retweeters' column)
    cascades['n_rts'] = cascades['retweeters'].str.count(' ')

    # Split the 'retweeters' column into 'user_id' of the original poster and 'retweeters'
    cascades[['user_id', 'retweeters']] = cascades['retweeters'].str.split(' ', 1, expand=True)

    # Read the followers network data from a CSV file, clean it, and create a 'user_id' column
    fn = pd.read_csv(data_path + 'followers_network.txt',
                     sep=' ', names=['followers'], header=None)
    fn.followers = fn.followers.str.replace(',', ' ')
    fn.followers = fn.followers.str.strip()
    fn['user_id'] = fn['followers'].str.split().str[0]    

    # Count the number of network followers for each user
    fn['n_network_followers'] = fn['followers'].str.count(' ')

    # Reset the index of the 'cascades' dataframe and merge it with the 'fn' dataframe based on 'user_id'
    cascades.reset_index(inplace=True)
    cascades = cascades.merge(fn, how='left', on='user_id')
    cascades = cascades.set_index('tweet_id')

    # Create a 'link' column by combining the Twitter URL and tweet IDs
    cascades['link'] = 'https://twitter.com/anyuser/status/' + cascades.index.values

    # Pick cascades with the highest n_network_followers, drop multiple cascades from a single user.
    cascades = cascades.sort_values(by='n_network_followers',ascending=False)[:5000].drop_duplicates(subset=['user_id'])

    # Filter out tweets with very few rts
    cascades = cascades[cascades.n_rts > 5]

    # Split 'retweeters,' 'followers,' and 'timestamps' columns to lists
    cascades.retweeters = cascades.retweeters.str.split()
    cascades.followers = cascades.followers.str.split()
    cascades.timestamps = cascades.timestamps.str.split()

    # Define a function to convert a list of strings to a list of floats
    def str_list_to_float_list(str_list):
        return [float(x) for x in str_list]

    # Apply the function to the 'timestamps' column
    cascades['timestamps'] = cascades['timestamps'].apply(str_list_to_float_list)
    
    
    # Scale timestamps so they start at 0 and signify hours.
    cascades['scaled_timestamps'] = cascades['timestamps'].apply(scale_timestamps)    

    # Merge with manually created full_text
    full_text = pd.read_csv(data_path + 'cascades.csv', index_col = 'tweet_id', usecols = ['tweet_id','text', 'n_followers','topic','natures'])
    full_text.index = full_text.index.astype('str')
    cascades = cascades.merge(full_text, left_index = True, right_index = True, how = 'left')

    cascades.index.astype('str')

    cascades.dropna(subset='text',inplace=True)

    cascades['length'] = cascades['text'].str.len()

    cascades['natures'] = cascades['natures'].astype(int)
    
    # This section is commented out to save OpenAI fees by not identifying the topics each time.
    """
    tweets = cascades.text
    topics = []
    for tweet in tweets:
        topic = ident_tweet_topic(tweet = tweet)
        topics.append(topic)
        print(topic)

    # Save topics list as a new col in cascades
    cascades['topic'] = topics
    """
    
    # Commented out to save OpenAI API fees. It marks it each tweet as factual (0), emotion (1), failed to classify (2)
    """
    tweets = cascades.text
    natures = []
    for tweet in tweets:
        nature = ident_fact_vs_emot(tweet = tweet)
        natures.append(nature)
        print(nature)

    natures_012 = ["2" if s not in {"0", "1"} else s for s in natures]
    natures_int = [int(x) for x in natures_012]
    print(natures_int)
    cascades['natures'] = natures_int
    cascades.to_csv(data_path + 'cascades.csv')
    """
    
    # Convert 'topic' column to a categorical data type
    cascades['topic'] = cascades['topic'].astype('category')

    # Define a mapping dictionary to merge categories
    category_mapping = {'Aftershocks and tremors': '3. Aftershocks and tremors', 
                    'Impact, aftermath, death toll': '2. Impact, aftermath, death toll', 
                    'Rescue operations and assistance': '1. Rescue operations and assistance',
                    'The tweet fits best with the topic "Faith and prayers."':'8. Faith and prayers',
                    'The tweet fits the topic of "Impact, aftermath, death toll."':'2. Impact, aftermath, death toll',
                    'The tweet fits the topic of "Technology, communication, and social media."':'5. Technology, communication, and social media',
                    'The tweet fits topic 5: Technology, communication, and social media.':'5. Technology, communication, and social media',
                    'Aftershocks and tremors': '3. Aftershocks and tremors',
                    'Relief efforts, resources, donations, and infrastructure':'7. Relief efforts, resources, donations, and infrastructure',
                    'The tweet fits the topic "Faith and prayers."':'8. Faith and prayers',
                    'The tweet fits the topic of "Relief efforts, resources, donations, and infrastructure."':'7. Relief efforts, resources, donations, and infrastructure' }
    
    # Use replace to merge categories based on the mapping
    cascades['topic'] = cascades['topic'].replace(category_mapping)
    
    # Create i_topic for Kappa test
    cascades['i_topic'] = cascades['topic'].str[0]
    cascades['i_topic'].replace('T', 10, inplace=True)

    # Create normalized n_rts
    cascades['n_rts_norm'] = cascades.n_rts / cascades.n_followers
    
    """# Create the MinMaxScaler
    scaler = MinMaxScaler()

    # Reshape the column to make it a 2D array
    cascades['n_rts_norm'] = cascades['n_rts_norm'].values.reshape(-1, 1)

    # Fit and transform the data
    cascades['n_rts_norm'] = scaler.fit_transform(cascades[['n_rts_norm']])"""
    
    groups = split_dataframe_by_category(cascades, 'topic')
    
    fn.followers = fn.followers.str.split() # NEW
    
    return cascades, fn, groups


def print_start_end(cascades):
    
    def convert_twitter_unix_timestamp_to_datetime(twitter_unix_timestamp):
        """
        Converts a Twitter Unix timestamp to a human-readable date and time.

        Args:
            twitter_unix_timestamp (int): The Twitter Unix timestamp to convert.

        Returns:
            str: A string representing the human-readable date and time in the format "YYYY-MM-DD HH:MM:SS".

        Example:
        >>> twitter_unix_timestamp = 1504649200
        >>> convert_twitter_unix_timestamp_to_datetime(twitter_unix_timestamp)
        '2017-09-05 12:20:00'
        """
        try:
            # Convert the Unix timestamp to a datetime object
            tweet_datetime = datetime.datetime.utcfromtimestamp(twitter_unix_timestamp)

            # Format the datetime as a string
            formatted_datetime = tweet_datetime.strftime("%Y-%m-%d %H:%M:%S")

            return formatted_datetime
        except ValueError:
            return "Invalid timestamp"
    
    # Dates of earliest and latest retweets in the dateset
    latest_date = convert_twitter_unix_timestamp_to_datetime(max(max(cascades.timestamps.values)))
    earliest_date = convert_twitter_unix_timestamp_to_datetime(min(min(cascades.timestamps.values)))
    print(f'Earliest date: {earliest_date}')
    print(f'Latest date: {latest_date}')
    return None



def draw_G(G, original_poster=None, followers=None):
    """
    Visualize a networkx graph with customized settings using draw_networkx.

    Parameters:
    G (networkx.Graph): The graph to be visualized.
    original_poster (str): The ID of the original poster node to be highlighted.
    retweeters (list): List of IDs of retweeter nodes.

    Returns:
    None
        The function displays the graph visualization.
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G) 
    
    #node_color = ['red' if node == original_poster else '#e2e2e2' if node in followers else '#808080' for node in G.nodes()]
    
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=False, node_size=200, node_color='#505050', font_size=8, font_color='black', alpha=0.3, width=0.8, arrowsize=6)
    
    if original_poster:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[original_poster], node_size=300, node_color='#ff0000', alpha = 1, label='Original Poster')
    
    if followers:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=followers, node_size=200, node_color='#000000', label='Followers', alpha = 1)

    return None

def remove_reflexive_edges(G):
    reflexive_edges = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(reflexive_edges)
    
    return G

"""def select_nodes_to_exc(G):
    # Find nodes with no outdegree
    no_outdegree_nodes = [node for node in G.nodes if G.out_degree(node) == 0]

    # Randomly sample half of the nodes with no outdegree
    sampled_nodes = random.sample(no_outdegree_nodes, k=int(0.97 * len(no_outdegree_nodes)))

    return sampled_nodes"""



def draw_G(G, original_poster=None, followers=None):
    """
    Visualize a networkx graph with customized settings using draw_networkx.

    Parameters:
    G (networkx.Graph): The graph to be visualized.
    original_poster (str): The ID of the original poster node to be highlighted.
    retweeters (list): List of IDs of retweeter nodes.

    Returns:
    None
        The function displays the graph visualization.
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G) 
    
    #node_color = ['red' if node == original_poster else '#e2e2e2' if node in followers else '#808080' for node in G.nodes()]
    
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=False, node_size=200, node_color='#505050', font_size=8, font_color='black', alpha=0.3, width=0.8, arrowsize=12)
    
    if original_poster:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[original_poster], node_size=300, node_color='#ff0000', alpha = 1, label='Original Poster')
    
    if followers:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=followers, node_size=200, node_color='#000000', label='Followers', alpha = 1)

    return None

def remove_reflexive_edges(G):
    reflexive_edges = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(reflexive_edges)
    
    return G




def select_nodes_to_exc(G, pct_exc = 0.97):
    # Find nodes with no outdegree
    no_outdegree_nodes = [node for node in G.nodes if G.out_degree(node) == 0]

    # Randomly sample half of the nodes with no outdegree
    sampled_nodes = random.sample(no_outdegree_nodes, k=int(pct_exc * len(no_outdegree_nodes)))

    return sampled_nodes



def draw_G(G, original_poster=None, followers=None):
    """
    Visualize a networkx graph with customized settings using draw_networkx.

    Parameters:
    G (networkx.Graph): The graph to be visualized.
    original_poster (str): The ID of the original poster node to be highlighted.
    retweeters (list): List of IDs of retweeter nodes.

    Returns:
    None
        The function displays the graph visualization.
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G) 
    
    #node_color = ['red' if node == original_poster else '#e2e2e2' if node in followers else '#808080' for node in G.nodes()]
    
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=False, node_size=200, node_color='#505050', font_size=8, font_color='black', alpha=0.3, width=0.8, arrowsize=12)
    
    if original_poster:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[original_poster], node_size=300, node_color='#ff0000', alpha = 1, label='Original Poster')
    
    if followers:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=followers, node_size=200, node_color='#000000', label='Followers', alpha = 1)

    return None

def remove_reflexive_edges(G):
    reflexive_edges = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(reflexive_edges)
    
    return G




def fit_null_model(X, y):
    model = LinearRegression()
    
    # Calculate pearson's r
    pearson_corr, p = pearsonr(X.flatten(), y) 
    
    # Calculate mean absolute residuals
    mad = np.mean(np.abs(model.fit(X, y).predict(X) - y))
    
    return model, mad, pearson_corr, p



def simulate_base_SI(cascades, beta, tweet_id, n_iter):
    """
    Simulates the spread of information in a social network using the mean-field Susceptible-Infectious (SI) model.

    Parameters:
    - beta (float): Transmission rate parameter, representing the probability of an susceptible individual
                   becoming infected when exposed to an infectious individual.
    - tweet_id (int): Identifier for the tweet in the social network cascade.
    - n_iter (int): Number of iterations to simulate the spread of the information.

    Returns:
    - tuple: A tuple containing the simulated number of infections at the end of the simulation (sim) 
             and the observed number of retweets for the given tweet (obs).

    The SI model assumes a simple spread of information in a social network where individuals are either
    susceptible (s) or infectious (i). The simulation progresses in discrete iterations, where at each
    iteration, the number of new infections is calculated based on the transmission rate (beta), the 
    current number of infectious individuals (i), and the number of susceptible individuals (s).

    Parameters:
    - s (float): Initial number of susceptible individuals, obtained from the network information
                associated with the given tweet_id.
    - i (float): Initial number of infectious individuals, set to 1 for the start of the simulation.

    Iteratively, for each iteration in the range of n_iter:
    - Calculate the number of new infections (infected) using the SI model equation: 
      infected = round(beta * i * s)
    - Update the number of susceptible individuals (s) by subtracting the newly infected individuals.
    - Update the number of infectious individuals (i) by adding the newly infected individuals.

    Finally, the function returns a tuple containing the simulated number of infections (sim) at the end
    of the simulation and the observed number of retweets (obs) for the given tweet_id from the cascades data.
    """
    s = cascades.loc[tweet_id].n_network_followers
    i = 1

    for iter in range(n_iter):
        # Calculate the number of new infections based on the SI model
        infected = np.round(beta * i * s)
        s = s - infected
        i = i + infected

    # Final simulation result and observed number of retweets for the given tweet
    sim = i
    obs = cascades.loc[tweet_id].n_rts
    
    return sim, obs



def fit_base_SI(data, n_iter):
    """
    Fits the mean-field Susceptible-Infectious (SI) model to observed data using a minimization approach.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing observed data for multiple tweets in a social network.
                               Each row corresponds to a tweet, and columns include relevant information
                               such as tweet identifiers, observed retweets, and network information.
    - n_iter (int): Number of iterations to simulate the spread of information in the SI model during the fitting process.

    Returns:
    - scipy.optimize.OptimizeResult: Result of the minimization process, containing information about the optimal
                                     transmission rate (beta) that minimizes the difference between simulated and
                                     observed data.

    The function uses a minimization approach to find the optimal transmission rate (beta) for the Basic Susceptible-
    Infectious (SI) model. It defines a loss function based on the Mean Absolute Error (MAE) between simulated and
    observed data for each tweet in the provided DataFrame.

    Parameters:
    - loss_function (function): Inner function that calculates the Mean Absolute Error (MAE) between simulated and
                                observed data for a given transmission rate (beta). It iterates over all tweet_ids
                                in the provided data and accumulates the absolute differences.
    - beta (float): Transmission rate parameter, representing the probability of a susceptible individual becoming
                   infected when exposed to an infectious individual.
    - tweet_ids (numpy.ndarray): Array containing unique identifiers for each tweet in the provided data.
    - sim (float): Simulated number of infections using the SI model for a specific tweet and transmission rate.
    - obs (float): Observed number of retweets for the corresponding tweet from the provided data.

    The `minimize_scalar` function from the scipy library is then used to find the optimal transmission rate that
    minimizes the total Mean Absolute Error (MAE) across all tweets. The result of the minimization process is
    returned as a scipy.optimize.OptimizeResult object.
    """
    def loss_function(beta):
        # Calculate the Mean Absolute Error (MAE) between simulated and observed data
        loss = 0
        tweet_ids = data.index.values
        for tweet_id in tweet_ids:
            sim, obs = simulate_base_SI(data, beta, tweet_id, n_iter)
            loss += abs(sim - obs)
        return loss

    # Use scalar minimization to find the optimal transmission rate (beta)
    result = minimize_scalar(loss_function, bounds=(0, 1))
    return result



def simulate_SI(cascades, G, beta, tweet_id, n_iter):
    """
    Simulates the spread of information in a social network using the Susceptible-Infectious (SI) model.

    Parameters:
    - G (networkx.Graph): The social network graph representing interactions between users.
    - beta (float): Transmission rate parameter, representing the probability of a susceptible individual
                   becoming infected when exposed to an infectious individual.
    - tweet_id (int): Identifier for the tweet in the social network cascade.
    - n_iter (int): Number of iterations to simulate the spread of the information.

    Returns:
    - tuple: A tuple containing the simulated number of infections at the end of the simulation (sim)
             and the observed number of retweets for the given tweet (obs).

    The SI model simulates the spread of information in a social network where individuals are either susceptible (S) or
    infectious (I). The simulation is performed using the NDlib library for Python. The function creates an instance
    of the SI model, configures it with the provided social network graph, sets the source node (user who tweeted the
    contagion), and runs the simulation for a specified number of iterations.

    Parameters:
    - model (ep.SIModel): Instance of the SI model representing the spread of information.
    - source_node (list): List containing the source node, which is the user who tweeted the contagion.
    - cfg (mc.Configuration): Configuration object specifying model parameters and initial infected nodes.
    - iterations (list): List of simulation iterations containing information about the status of nodes at each iteration.
                        Each iteration includes the count of susceptible and infected nodes.
    - sim (int): Simulated number of infections at the end of the simulation obtained from the final iteration.
    - obs (int): Observed number of retweets for the given tweet obtained from the 'cascades' DataFrame.

    The function returns a tuple containing the simulated number of infections (sim) at the end of the simulation and the
    observed number of retweets (obs) for the given tweet_id from the 'cascades' DataFrame.
    """
    # Create an SI model instance using the provided graph
    model = ep.SIModel(G)

    # Specify the source node as the user who tweeted the contagion
    source_node = [cascades.loc[tweet_id, 'user_id']]

    # Configure the SI model with the given beta value
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)

    # Set the initial configuration of infected nodes
    cfg.add_model_initial_configuration("Infected", source_node)
    model.set_initial_status(cfg)

    # Run the simulation for the specified number of iterations
    iterations = model.iteration_bunch(n_iter)

    # Extract the simulated number of retweets from the iterations
    sim = iterations[-1]['node_count'][1]
    # Retrieve the observed number of retweets from the 'cascades' DataFrame
    obs = cascades.loc[tweet_id, 'n_rts']

    return sim, obs



def fit_SI(data, n_iter, fn):
    """
    Fits the Susceptible-Infectious (SI) model to observed data using a minimization approach.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing observed data for multiple tweets in a social network.
                               Each row corresponds to a tweet, and columns include relevant information
                               such as tweet identifiers, observed retweets, and network information.
    - n_iter (int): Number of iterations to simulate the spread of information in the SI model during the fitting process.

    Returns:
    - float: Optimal transmission rate (beta) that minimizes the Mean Absolute Error (MAE) between simulated and
             observed data.

    The function uses a minimization approach to find the optimal transmission rate (beta) for the Susceptible-
    Infectious (SI) model. It defines a loss function based on the Mean Absolute Error (MAE) between simulated and
    observed data for each tweet in the provided DataFrame.

    Parameters:
    - loss_function (function): Inner function that calculates the Mean Absolute Error (MAE) between simulated and
                                observed data for a given transmission rate (beta). It iterates over all tweet_ids
                                in the provided data and accumulates the absolute differences.
    - beta (float): Transmission rate parameter, representing the probability of a susceptible individual becoming
                   infected when exposed to an infectious individual.
    - tweet_ids (numpy.ndarray): Array containing unique identifiers for each tweet in the provided data.
    - G (networkx.Graph): Social network graph representing interactions between users in the cascade.
    - sim (int): Simulated number of infections using the SI model for a specific tweet and transmission rate.
    - obs (int): Observed number of retweets for the corresponding tweet from the provided data.

    The `minimize_scalar` function from the scipy library is then used with the bounded optimization method to find
    the optimal transmission rate that minimizes the total Mean Absolute Error (MAE) across all tweets. The result
    is the optimal beta value.
    """
    
    def loss_function(beta):
        loss = 0
        tweet_ids = data.index.values
        for tweet_id in tweet_ids:
            G = create_G(data, tweet_id, fn)
            sim, obs =  simulate_SI(data, G, beta, tweet_id, n_iter)
            loss += abs(obs - sim)
        return loss

    # Use a simple optimization method to find the beta that minimizes the MSE
    result = minimize_scalar(loss_function, bounds = (0, 1), method = 'bounded')
    return result.x



def simulate_IC(cascades, G, threshold, tweet_id, n_iter):
    """
    Simulates the spread of information in a social network using the Independent Cascades (IC) model.

    Parameters:
    - G (networkx.Graph): The social network graph representing interactions between users.
    - threshold (float): Activation threshold parameter for the Independent Cascades model,
                        representing the probability of an edge transmitting the information.
    - tweet_id (str): Identifier for the tweet in the social network cascade.
    - n_iter (int): Number of iterations to simulate the spread of the information.

    Returns:
    - tuple: A tuple containing the simulated number of infections at the end of the simulation (sim)
             and the observed number of retweets for the given tweet (obs).

    The Independent Cascades (IC) model simulates the spread of information in a social network, where nodes are either
    infected or not infected. The simulation is performed using the NDlib library for Python. The function creates
    an instance of the Independent Cascades model, configures it with the provided social network graph, sets the
    source node (user who tweeted the contagion), and runs the simulation for a specified number of iterations.

    Parameters:
    - model (ep.IndependentCascadesModel): Instance of the NDlib Independent Cascades model representing
                                           the spread of information.
    - source_node (list): List containing the source node, which is the user who tweeted the contagion.
    - cfg (mc.Configuration): Configuration object specifying model parameters and initial infected nodes.
    - iterations (list): List of simulation iterations containing information about the status of nodes at each iteration.
                        Each iteration includes the count of infected and not infected nodes.
    - sim (int): Simulated number of infections at the end of the simulation obtained from the final iteration.
    - obs (int): Observed number of retweets for the given tweet obtained from the 'cascades' DataFrame.

    The function returns a tuple containing the simulated number of infections (sim) at the end of the simulation and the
    observed number of retweets (obs) for the given tweet_id from the 'cascades' DataFrame.
    """
    # Model selection
    model = ep.IndependentCascadesModel(G)

    # Infer source node from tweet_id by looking up the cascades df. Put it in a list
    source_node = [cascades.loc[tweet_id,'user_id']]
    
    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_initial_configuration("Infected", source_node)

    # Setting the edge parameters
    for e in G.edges():
        cfg.add_edge_configuration("threshold", e, threshold)
        
    model.set_initial_status(cfg)

    # Simulation execution
    iterations = model.iteration_bunch(n_iter)
    
    # Prepare your outputs
    sim = iterations[-1]['node_count'][1] + iterations[-1]['node_count'][2]
    obs = cascades.loc[tweet_id, 'n_rts']
    
    return sim, obs




def fit_IC(data, n_iter, fn):
    """
    Fits the Independent Cascades (IC) model to observed data using a minimization approach.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing observed data for multiple tweets in a social network.
                               Each row corresponds to a tweet, and columns include relevant information
                               such as tweet identifiers, observed retweets, and network information.
    - n_iter (int): Number of iterations to simulate the spread of information in the IC model during the fitting process.

    Returns:
    - float: Optimal activation threshold that minimizes the Mean Absolute Deviation (MAD) between simulated and
             observed data.

    The function uses a minimization approach to find the optimal activation threshold for the Independent Cascades (IC)
    model. It defines a loss function based on the Mean Absolute Deviation (MAD) between simulated and observed data for
    each tweet in the provided DataFrame.

    Parameters:
    - loss_function (function): Inner function that calculates the Mean Absolute Deviation (MAD) between simulated and
                                observed data for a given activation threshold. It iterates over all tweet_ids in the
                                provided data and accumulates the absolute differences.
    - threshold (float): Activation threshold parameter for the Independent Cascades model, representing the probability
                        of an edge transmitting the information.
    - tweet_ids (numpy.ndarray): Array containing unique identifiers for each tweet in the provided data.
    - G (networkx.Graph): Social network graph representing interactions between users in the cascade.
    - sim (int): Simulated number of infections using the IC model for a specific tweet and activation threshold.
    - obs (int): Observed number of retweets for the corresponding tweet from the provided data.

    The `minimize_scalar` function from the scipy library is then used with the bounded optimization method to find
    the optimal activation threshold that minimizes the total Mean Absolute Deviation (MAD) across all tweets. The result
    is the optimal threshold value.
    """
    def loss_function(threshold):
        # Calculate the Mean Absolute Deviation (MAD) between simulated and observed data
        loss = 0
        tweet_ids = data.index.values
        for tweet_id in tweet_ids:
            G = create_G(data, tweet_id, fn)
            sim, obs = simulate_IC(data, G, threshold, tweet_id, n_iter)
            loss += abs(sim - obs)
        mad = loss / len(tweet_ids)  # Calculate MAD here
        return loss

    # Use scalar minimization to find the optimal activation threshold
    result = minimize_scalar(loss_function, method='bounded', bounds=(0.0001, 0.1))
    return result.x


def exponential_decay(A, k, t):
    """
    Calculates the value of an exponentially decaying function at a given time.

    Parameters:
    - A (float): The initial amplitude or value of the function at time t=0.
    - k (float): The decay constant, determining the rate of decay.
    - t (float): Time at which to evaluate the function.

    Returns:
    - float: The value of the exponentially decaying function at the specified time.

    The exponential decay function is given by the formula: A * exp(-k * t), where:
    - A is the initial amplitude or value of the function at time t=0.
    - k is the decay constant, determining the rate of decay.
    - t is the time at which to evaluate the function.

    The function uses the math.exp() function from the math module to calculate the exponential function.

    Example:
    >>> exponential_decay(10, 0.1, 2)
    Output: 6.737947005
    """
    return A * math.exp(-k * t)


def sim_decaying_IC(cascades, G, threshold, tweet_id, n_iter, k):
    """
    Simulates the spread of information in a social network using the Independent Cascades (IC) model with a decaying threshold.

    Parameters:
    - G (networkx.Graph): The social network graph representing interactions between users.
    - threshold (float): Initial activation threshold parameter for the Independent Cascades model,
                        representing the probability of an edge transmitting the information at time t=0.
    - tweet_id (str): Identifier for the tweet in the social network cascade.
    - n_iter (int): Number of iterations to simulate the spread of information in the IC model with decaying threshold.
    - k (float): Decay constant, determining the rate at which the threshold decays.

    Returns:
    - tuple: A tuple containing the simulated number of infections at the end of the simulation (sim)
             and the observed number of retweets for the given tweet (obs).

    The function simulates the spread of information in a social network using the Independent Cascades (IC) model,
    where nodes are either infected or not infected. The activation threshold for edges decays exponentially over
    iterations with the specified decay constant (k).

    Parameters:
    - model (ep.IndependentCascadesModel): Instance of the NDlib Independent Cascades model representing
                                           the spread of information.
    - source_node (list): List containing the source node, which is the user who tweeted the contagion.
    - cfg (mc.Configuration): Configuration object specifying model parameters and initial infected nodes.
    - iterations (list): List of simulation iterations containing information about the status of nodes at each iteration.
                        Each iteration includes the count of infected and not infected nodes.
    - threshold (float): Activation threshold parameter for the Independent Cascades model at each iteration,
                        decaying exponentially with time.
    - sim (int): Simulated number of infections at the end of the simulation obtained from the final iteration.
    - obs (int): Observed number of retweets for the given tweet obtained from the 'cascades' DataFrame.

    The function returns a tuple containing the simulated number of infections (sim) at the end of the simulation and the
    observed number of retweets (obs) for the given tweet_id from the 'cascades' DataFrame.
    """
    # Model selection
    model = ep.IndependentCascadesModel(G)

    # Infer source node from tweet_id by looking up the cascades df. Put it in a list
    source_node = [cascades.loc[tweet_id, 'user_id']]

    # Model Configuration
    cfg = mc.Configuration()

    # Set the initial infected node
    cfg.add_model_initial_configuration("Infected", source_node)

    # Set initial status
    model.set_initial_status(cfg)

    # Initialize iterations to be able to append to it later on
    iterations = []

    for i in range(n_iter - 1):
        # Update threshold for all edges in cfg and update the model
        for e in G.edges():
            cfg.add_edge_configuration("threshold", e, threshold)
        model.set_initial_status(cfg)

        # Execute the model
        iterations.append(model.iteration_bunch(1))

        # Decay threshold
        threshold = exponential_decay(A=threshold, k=k, t=i)

        # Prepare your func outputs
        sim = iterations[-1][0]['node_count'][1] + iterations[-1][0]['node_count'][2]
        obs = cascades.loc[tweet_id, 'n_rts']

    return sim, obs, iterations



def fit_decay_param(data, n_iter, threshold, fn):
    """
    Fits the Independent Cascades (IC) model with a decaying threshold to observed data using a minimization approach.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing observed data for multiple tweets in a social network.
                               Each row corresponds to a tweet, and columns include relevant information
                               such as tweet identifiers, observed retweets, and network information.
    - n_iter (int): Number of iterations to simulate the spread of information in the IC model with decaying threshold
                    during the fitting process.
    - threshold (float): Initial activation threshold parameter for the Independent Cascades model at time t=0,
                        representing the probability of an edge transmitting the information.

    Returns:
    - scipy.optimize.OptimizeResult: Result of the minimization process, containing information about the optimal
                                     decay constant (k) that minimizes the Mean Absolute Deviation (MAD) between
                                     simulated and observed data.

    The function uses a minimization approach to find the optimal decay constant (k) for the Independent Cascades (IC)
    model with a decaying threshold. It defines a loss function based on the Mean Absolute Deviation (MAD) between
    simulated and observed data for each tweet in the provided DataFrame.

    Parameters:
    - loss_function (function): Inner function that calculates the Mean Absolute Deviation (MAD) between simulated and
                                observed data for a given decay constant (k). It iterates over all tweet_ids in the
                                provided data and accumulates the absolute differences.
    - k (float): Decay constant parameter, determining the rate at which the threshold decays.
    - tweet_ids (numpy.ndarray): Array containing unique identifiers for each tweet in the provided data.
    - G (networkx.Graph): Social network graph representing interactions between users in the cascade.
    - sim (int): Simulated number of infections using the IC model with a decaying threshold for a specific tweet and decay constant.
    - obs (int): Observed number of retweets for the corresponding tweet from the provided data.

    The `minimize_scalar` function from the scipy library is then used to find the optimal decay constant (k) that
    minimizes the total Mean Absolute Deviation (MAD) across all tweets. The result of the minimization process is
    returned as a scipy.optimize.OptimizeResult object.
    """
    def loss_function(k):
        # Calculate the Mean Absolute Deviation (MAD) between simulated and observed data
        loss = 0
        tweet_ids = data.index.values
        for tweet_id in tweet_ids:
            G = create_G(data, tweet_id, fn)
            sim, obs, _ = sim_decaying_IC(data, G, threshold, tweet_id, n_iter, k)
            loss += abs(sim - obs)
        return loss

    # Use scalar minimization to find the optimal decay constant (k)
    result = minimize_scalar(loss_function)
    return result


def pick_n_tweets(cascades, n, seed):
    """
    Selects and returns a list of unique tweet IDs from a collection of tweet cascades.

    Parameters:
    - n (int): The number of tweet IDs to pick.
    - seed (int): Seed for reproducibility, ensures consistent random selections.

    Returns:
    - list: A list containing 'n' unique tweet IDs randomly chosen from the tweet cascades.

    Usage Example:
    --------------
    tweet_ids = pick_n_tweets(n=15, seed=8)

    Description:
    ------------
    This function is designed to randomly select 'n' unique tweet IDs from a predefined collection of tweet cascades.
    The randomization is performed using NumPy's random module, with the provided 'seed' ensuring reproducibility.
    
    The tweet IDs are selected using the internal function 'pick_random_tweet_id' with the collection of cascades as
    an input parameter. The selected tweet IDs are then appended to a list and returned.

    Note:
    -----
    It is assumed that the 'pick_random_tweet_id' function is defined elsewhere and takes the tweet cascades as an
    argument, returning a randomly selected tweet ID from the provided collection.

    Parameters and their Descriptions:
    -----------------------------------
    - n (int): The number of tweet IDs to be randomly selected and returned by the function.
    - seed (int): An integer seed for the random number generator to ensure reproducibility. By providing the same
      seed value, the function will yield the same set of random tweet IDs on subsequent runs.

    Returns:
    --------
    - list: A list containing 'n' unique tweet IDs randomly chosen from the given tweet cascades.

    Example Usage:
    --------------
    tweet_ids = pick_n_tweets(n=15, seed=8)
    # This will generate a list of 15 unique tweet IDs randomly selected from the cascades.

    """
    np.random.seed(seed)
    tweet_ids = []
    for _ in range(n):
        tweet_ids.append(pick_random_tweet_id(cascades))
    return tweet_ids



def create_list_cum_n_rts(cascades, fn, tweet_ids, jitter_factor):
    """
    Generate a list of cumulative retweet counts over iterations for each tweet in the given list of tweet IDs.

    Parameters:
    - tweet_ids (list): A list of tweet IDs for which the cumulative retweet counts are to be calculated.

    Returns:
    - list_cum_n_rts_scaled (list): A list containing scaled cumulative retweet counts for each tweet.
    
    Algorithm:
    1. For each tweet_id in the input list, create a cascade network G using the provided cascades and tweet_id.
    2. Calculate cumulative retweet counts over iterations using the sim_decaying_IC function.
    3. Extract node counts for each iteration and store them in a list.
    4. Apply Min-Max scaling to each sublist of cumulative retweet counts.
    5. Return the list of scaled cumulative retweet counts.

    Note:
    - The scaling is performed using Min-Max scaling from the scikit-learn library.
    - The sim_decaying_IC function and create_G function are assumed to be defined elsewhere in the codebase.

    See Also:
    - sim_decaying_IC: Function for simulating information cascades with decaying influence.
    - create_G: Function for creating a network graph for a given cascade and tweet_id.
    """
    list_cum_n_rts = []

    # Iterate over each tweet_id
    for tweet_id in tweet_ids:
        # Create cascade network G for the current tweet_id
        G = create_G(cascades, tweet_id, fn)
        
        # Simulate information cascades and get cumulative retweet counts over iterations
        sim, obs, iterations = sim_decaying_IC(cascades, G, threshold=0.003, tweet_id=tweet_id, n_iter=5, k=0.94)

        # Extract node counts for each iteration and store in a list
        iter_0 = iterations[0][0]['node_count'][1] + iterations[0][0]['node_count'][2]
        iter_1 = iterations[1][0]['node_count'][1] + iterations[1][0]['node_count'][2]
        iter_2 = iterations[2][0]['node_count'][1] + iterations[2][0]['node_count'][2]
        iter_3 = iterations[3][0]['node_count'][1] + iterations[3][0]['node_count'][2]
        list_cum_n_rts.append([iter_0, iter_1, iter_2, iter_3])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply the min-max scaling to each sublist
    list_cum_n_rts_scaled = [scaler.fit_transform(np.array(sublist).reshape(-1, 1)).flatten().tolist() for sublist in list_cum_n_rts]
    
    list_cum_n_rts_jittered = [[val + np.random.uniform(-jitter_factor, jitter_factor) for val in sublist] for sublist in list_cum_n_rts_scaled]

    return list_cum_n_rts_jittered



def fit_log_model_iter(data):
    
    # Subset your data, create y variable
    y = data[1:]
    x = [1,2,3]
    
    # Fit the logarithmic model to the data
    params, covariance = curve_fit(logarithmic_function, x, y)
    
    # Extract the estimated parameters
    a, b = params
    
    return params



def calc_mean_error_per_cascade(timestamps):
    """
    Calculates the mean absolute error between predicted and true values for a set of timestamps.

    Parameters:
    - timestamps (numpy.ndarray or list): Array or list containing timestamp values.

    Returns:
    - float: Mean absolute error between predicted and true values.

    The function generates predicted values using a logarithmic function and calculates the mean absolute error
    between the predicted and true values.

    Parameters:
    - timestamps (numpy.ndarray or list): Array or list of timestamp values for which to calculate the mean absolute error.
    - y_true (numpy.ndarray): Array containing true values, generated as a linspace from 0 to 1.
    - y_pred (numpy.ndarray): Predicted values generated using a logarithmic function.
    - mean_abs_err (float): Mean absolute error calculated between y_true and y_pred.

    Example:
    >>> timestamps = [0, 2, 4, 6, 8, 10]
    >>> calc_mean_error_per_cascade(timestamps)
    Output: 0.123
    """
    y_true = np.linspace(0, 1, len(timestamps))[1:]
    y_pred = logarithmic_function(a=0.24, b=0.7, x=y_true)

    mean_abs_err = np.mean(abs(y_pred - y_true))

    return mean_abs_err

def calc_grand_mean_abs_err(list_cum_n_rts_scaled):
    mean_errors = []
    for cum_n_rts_scaled in list_cum_n_rts_scaled:
        mean_error = calc_mean_error_per_cascade(cum_n_rts_scaled)
        mean_errors.append(mean_error)
        grand_mean_abs_err = np.mean(mean_errors)
    return grand_mean_abs_err


def plot_timestamps_ecdf(timestamps):
    
    """
    Plots the Empirical Cumulative Distribution Function (ECDF) of timestamps.

    Parameters:
    - timestamps (numpy.ndarray or list): Array or list containing timestamp values.

    Returns:
    - None

    The function scales the timestamps, creates an ECDF plot, and visualizes the cumulative proportion of retweets
    over time.

    Parameters:
    - timestamps (numpy.ndarray or list): Array or list of timestamp values to be plotted.

    The function uses the seaborn library to create an ECDF plot with markers for each timestamp. The plot includes
    labels for the x and y-axes, a title, and a legend. The y-axis limits are set to ensure the top marker is not cut-off.
    The top and right spines are removed for a cleaner appearance.

    Example:
    >>> timestamps = [0, 2, 4, 6, 8, 10]
    >>> plot_timestamps_ecdf(timestamps)
    """

    timestamps = scale_timestamps(timestamps)
    
    # Close any previous figure to avoid interference
    plt.close()
    
    # Set the figure size to 16:9 aspect ratio
    plt.figure(figsize=(16/2, 9/2))
  
    # Create an Empirical Cumulative Distribution Function (ECDF) plot
    sns.ecdfplot(timestamps, marker="o", linestyle="-", alpha = 1, markersize = 5, markerfacecolor='None',markeredgecolor='gray', color = (0.5,0.5,0.5))
    
    # Add labels and title to the plot
    plt.xlabel('Hours passed since the original tweet',labelpad=20)
    plt.ylabel('Proportion of retweets',labelpad=20)
    plt.title('Cumulative proportion of tweets shared over time',pad=20)
    plt.legend(['Retweet'],loc='lower right')
    
    # Set y-axis limits to ensure the top marker is not cut-out
    plt.ylim(0, 1.06)
    
    # Remove the top and right spines for a cleaner appearance
    sns.despine()
    
    # Save the plot as an SVG file
    plt.savefig(fig_path + 'ecdf_plot.svg', format='svg', bbox_inches='tight')

    # Display the plot
    plt.show()



def avrami_equation(t, k, n):
    """
    Calculate the Avrami equation value at a given time.

    The Avrami equation is a mathematical model used to describe phase transformation
    kinetics in materials, such as crystallization and nucleation. It is expressed
    as:

    A(t) = 1 - exp(-k * t^n)

    where:
    - A(t) is the fraction of transformation at time t.
    - k is the rate constant that depends on the specific process.
    - n is the Avrami exponent that characterizes the reaction mechanism.

    Parameters:
    t (float): Time at which to calculate the transformation fraction.
    k (float): Rate constant of the Avrami equation.
    n (float): Avrami exponent, characterizing the reaction mechanism.

    Returns:
    float: The fraction of transformation at time t according to the Avrami equation.

    Example:
    >>> avrami_equation(2.5, 0.1, 2)
    0.18126924692201818
    """
    return 1 - np.exp(-k * t**n)


def plot_avrami_equation(k, n, t_max=48, num_points=100):
    """
    Plots the Avrami equation for a given set of parameters.

    The Avrami equation is used to model phase transformation kinetics in materials,
    describing how the fraction transformed changes over time during processes like
    crystallization, precipitation, etc.

    Parameters:
    k (float): The rate constant in the Avrami equation.
    n (float): The Avrami exponent in the Avrami equation.
    t_max (float, optional): The maximum time for the plot. Defaults to 10.
    num_points (int, optional): The number of points to generate in the time range.
        Defaults to 100.

    Returns:
    None: This function plots the Avrami equation but does not return any value.

    Example:
    plot_avrami_equation(k=0.1, n=2)

    This will generate a plot of the Avrami equation with the specified k and n values.
    You can adjust the t_max and num_points parameters to control the time range and
    the number of points in the plot.
    """
    t = np.linspace(0, t_max, num_points)
    f = avrami_equation(t, k, n)
    
    plt.close()
    plt.figure(figsize=(16/2, 9/2))
    plt.plot(t, f, label=f'k={k}, n={n}',color='gray')
    plt.xlabel('Time (t)', labelpad = 20)
    plt.ylabel('Proportion of retweets f(t)', labelpad = 20)
    plt.title('Avrami Equation Plot',pad = 20)
    plt.legend(loc = 'lower right')
    sns.despine()
    plt.savefig('/Users/ekinderdiyok/Documents/Thesis/Figures/avrami.svg', bbox_inches='tight')
    plt.savefig('/Users/ekinderdiyok/Documents/Thesis/Figures/avrami.png', dpi = 400,  bbox_inches='tight')
    plt.show()



def logarithmic_function(x, a, b):
    return a * np.log(x) + b



def fit_log_model(data):
    
    # Subset your data, create y variable
    x = data[1:]
    y = np.linspace(0, 1, len(data))[1:]
    
    # Fit the logarithmic model to the data
    params, covariance = curve_fit(logarithmic_function, x, y)
    
    # Extract the estimated parameters
    a, b = params

    #print(f'a * log(x) + b: parameter a is {a}, b is {b})')
    
    return params




def calc_mean_error_per_cascade(timestamps):
    """
    Calculates the mean absolute error between predicted and true values for a set of timestamps.

    Parameters:
    - timestamps (numpy.ndarray or list): Array or list containing timestamp values.

    Returns:
    - float: Mean absolute error between predicted and true values.

    The function generates predicted values using a logarithmic function and calculates the mean absolute error
    between the predicted and true values.

    Parameters:
    - timestamps (numpy.ndarray or list): Array or list of timestamp values for which to calculate the mean absolute error.
    - y_true (numpy.ndarray): Array containing true values, generated as a linspace from 0 to 1.
    - y_pred (numpy.ndarray): Predicted values generated using a logarithmic function.
    - mean_abs_err (float): Mean absolute error calculated between y_true and y_pred.

    Example:
    >>> timestamps = [0, 2, 4, 6, 8, 10]
    >>> calc_mean_error_per_cascade(timestamps)
    Output: 0.123
    """
    y_true = np.linspace(0, 1, len(timestamps))[1:]
    y_pred = logarithmic_function(a=0.17, b=0.77, x=y_true)

    mean_abs_err = np.mean(abs(y_pred - y_true))

    return mean_abs_err



"""def calc_grand_mean_abs_err(data):
    mean_errors = []
    tweet_ids = data.index.values
    for tweet_id in tweet_ids:
        mean_error = calc_mean_error_per_cascade(cascades.loc[tweet_id].timestamps)
        mean_errors.append(mean_error)
        grand_mean_abs_err = np.mean(mean_errors)
    return grand_mean_abs_err"""

