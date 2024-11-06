import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import sys

sys.path.append('./')
import glob
from statistics import mean

from evaluators import load_evaluator
from needle_config import needle_dict

def load_as_df(folder_path, re_evaluation_method=None, needle_name='SF', verbose=False):
    json_files = glob.glob(f"{folder_path}/*.json")

    # List to hold the data
    data = []
    
    if re_evaluation_method is not None:
        evaluator = load_evaluator(re_evaluation_method, substr_validation_words=needle_dict[needle_name]['substr_validation_words'].split(','))

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)

            if re_evaluation_method is not None:
                model_response = json_data["model_response"]

                score = evaluator.evaluate_response(model_response)
            else:
                score = json_data.get("score", None)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })
    
    # Creating a DataFrame
    df = pd.DataFrame(data)

    if verbose:
        print(df.head())
        print(f"You have {len(df)} rows")

    return df

def create_pivot_table(df, verbose=False, min_length=0, max_length=2350):
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], 
                                 aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table[(pivot_table['Context Length'] <= max_length) & (pivot_table['Context Length'] >= min_length)]
    acc = mean(pivot_table['Score'])
    print(f"Accuracy: {acc}")

    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", 
                                    values="Score") # This will turn into a proper pivot
    if verbose:
        pivot_table.iloc[:5, :5]

    return pivot_table

def plot_pivot_table(pivot_table, model_name="", save_path=None, verbose=False):
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'}
    )
    
    # More aesthetics
    plt.title(f'Pressure Testing {model_name}\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    if verbose:
        # Show the plot
        plt.show()

def main(folder_path: str, save_path: str=None, needle_name: str="SF", verbose: bool=False, min_length=0, max_length=2250, re_evaluation_method: str=None):
    model_name = folder_path.strip('/').split('/')[-1].split("_")[0]
    if save_path is None:
        save_path = f"./{folder_path.strip('/').split('/')[-1]}.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # if os.path.isfile(save_path):
        #     raise ValueError(f"{save_path} already exists.")
    df = load_as_df(folder_path, re_evaluation_method=re_evaluation_method, needle_name=needle_name, verbose=verbose)
    pivot_table = create_pivot_table(df, verbose=verbose, min_length=min_length, max_length=max_length)
    plot_pivot_table(pivot_table, model_name=model_name, save_path=save_path, verbose=verbose)

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)
