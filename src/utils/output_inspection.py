import pandas as pd
import os
from src.debug.debug_general import print_dataframe
import ast
import matplotlib.pyplot as plt

def analyze_csv(csv_path: str, plot_loss_charts: bool = False) -> None:
    """
    Analyze the results from a CSV file containing grid search data, extracting tested parameters and
    best parameters based on avg_iou_accuracy, optionally plotting loss trends.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing the grid search results.
    plot_loss_charts : bool, optional
        Whether to plot the loss charts.


    """
    parameter_mapping = {
        "gs_nn_film_layers": "nn_film_layers",
        "gs_nn_scheduler": "nn_use_scheduler",
        "gs_nn_scheduler_power": "nn_scheduler_power",
        "gs_nn_grad_clipping": "nn_grad_clipping",
        "gs_nn_optimizer": "nn_optimizer",
        "gs_nn_dropout_prob": "nn_dropout_prob",
        "gs_nn_opt_learning_rate": "nn_opt_learning_rate",
        "gs_nn_bbox_hidden_dim": "nn_bbox_hidden_dim",
        "gs_nn_wd": "nn_wd",
        "gs_nn_giou_lambda": "nn_giou_lambda",
        "gs_nn_l1_lambda": "nn_l1_lambda",
        "gs_nn_bce_lambda": "nn_bce_lambda",
        "gs_dt_extra_similar_sents" : "Extra Similar Sentences",
        "gs_dt_apply_template" : "Apply templates",
        "gs_dt_ess_sm_k" : "Similarity Matrix K",
        "gs_bs_supp_model" : "Baseline Obj. Det. Model",
        "gs_bs_sents_vector_type" : "Sentences Vector Type"
        }

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    
    df = pd.read_csv(csv_path)

    # Extract tested parameters
    tested_parameters = {}
    for gs_param, csv_param in parameter_mapping.items():
        if csv_param in df.columns:
            tested_parameters[gs_param] = df[csv_param].dropna().unique().tolist()
        
    # Extract the best parameters
    if "avg_iou_accuracy" not in df.columns:
        raise ValueError("The column 'avg_iou_accuracy' is not present in the CSV file.")
    
    best_row = df.loc[df["avg_iou_accuracy"].idxmax()]
    best_parameters = {}
    for gs_param, csv_param in parameter_mapping.items():
        if csv_param in best_row.index:
            best_parameters[gs_param] = best_row[csv_param]

    print("Parameters values tested:\n")
    for param, values in tested_parameters.items():
        print(f"{param}: {values}")
    
    print("\nBest parameters configuration based on avg_iou_accuracy\n")
    for param, value in best_parameters.items():
        print(f"{param}: {value}")

    df = df.sort_values(by="avg_iou_accuracy", ascending=False)
    print(f"\nTop 5 combinations:\n")
    
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    df['avg_iou_accuracy'] = pd.to_numeric(df['avg_iou_accuracy'], errors='coerce')
    df = df.sort_values(by='avg_iou_accuracy', ascending=False)
    
    if plot_loss_charts:
        plt.figure(figsize=(10, 10))
        
        # Extract losses and plot them
        for idx, row in df[:10].iterrows():
            losses_list = ast.literal_eval(row['train_losses'])
            plt.plot(losses_list, label=f"Exp ID: {row['exp_id']}")

        plt.title(f"Loss Trends")
        plt.xlabel('Iterations')
        plt.ylabel('Loss Value')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Show the plot
        plt.show()
            
    return df

def show_grid_search_results(csv_file_path: str, topk: int = 3) -> None:
    """
    Inspect the baseline grid search output from a CSV file, showing the top-k experiments based on 
    average IoU accuracy and the relevant columns.

    Parameters
    ----------
    csv_file_path : str
        The path to the CSV file containing the grid search results.
    topk : int, optional
        The number of top experiments to display.

    """
    column_names = {
        "dt_splits": "Dataset Splits",
        "dt_samples_limit": "Samples Limit",
        "dt_templates": "Templates",
        "dt_apply_template": "Apply Template",
        "dt_extra_similar_sents": "Extra Similar Sentences",
        "dt_ess_sm_k": "Similarity Matrix k",
        "dt_sents_vector_type": "Sentence Vector Type",
        "bs_supp_model": "Baseline Model",
        "bs_sents_vector_type": "Baseline Sent. Vector Type",
        "avg_iou_accuracy": "Average IoU Accuracy",
    }
    
    pd.set_option('display.float_format', '{:.6f}'.format)
    df = pd.read_csv(csv_file_path)
    df['avg_iou_accuracy'] = pd.to_numeric(df['avg_iou_accuracy'], errors='coerce')
    df = df.sort_values(by='avg_iou_accuracy', ascending=False)
    df.rename(columns=column_names, inplace=True)
    print_dataframe(df, limit=topk)
