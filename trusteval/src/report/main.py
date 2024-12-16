import pandas as pd
from jinja2 import Environment, FileSystemLoader
import os
from .case_generator import generate_case, return_dict  # Add return_dict to the import
import markdown  # Add this import
import glob  # Add this import
import shutil  # Add this import
import sys

# Set template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_HTML = "report.html"
sys.path.append(PROJECT_ROOT)
from trusteval.src.generation import ModelService

service = ModelService(
    request_type='llm',
    handler_type='api',
    model_name='gpt-4o',
    config_path=os.path.join(PROJECT_ROOT, 'src/config/config.yaml'),
    temperature=0.6
)

def load_model_info():
    """
    Loads model information from the provided table.
    """
    model_info = {
        'gpt-4o': {
            'model_name': 'GPT-4o',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'OpenAI',
            'version': '2024-08-06'
        },
        'gpt-4o-mini': {
            'model_name': 'GPT-4o-mini',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'OpenAI',
            'version': '2024-07-18'
        },
        'gpt-3.5-turbo': {
            'model_name': 'GPT-3.5-Turbo',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'OpenAI',
            'version': '0125'
        },
        'claude-3.5-sonnet': {
            'model_name': 'Claude-3.5-Sonnet',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'Anthropic',
            'version': '20240620'
        },
        'claude-3-haiku': {
            'model_name': 'Claude-3-Haiku',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'Anthropic',
            'version': '20240307'
        },
        'gemini-1.5-pro': {
            'model_name': 'Gemini-1.5-Pro',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'Google',
            'version': '002'
        },
        'gemini-1.5-flash': {
            'model_name': 'Gemini-1.5-Flash',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'Google',
            'version': '002'
        },
        'gemma-2-27b': {
            'model_name': 'Gemma-2-27B',
            'model_size': '27B',
            'open_weight': True,
            'creator': 'Google',
            'version': 'it'
        },
        'llama-3.1-70b': {
            'model_name': 'Llama-3.1-70B',
            'model_size': '70B',
            'open_weight': True,
            'creator': 'Meta',
            'version': 'instruct'
        },
        'llama-3.1-8b': {
            'model_name': 'Llama-3.1-8B',
            'model_size': '8B',
            'open_weight': True,
            'creator': 'Meta',
            'version': 'instruct'
        },
        'mixtral-8*22b': {
            'model_name': 'Mixtral-8*22B',
            'model_size': '8*22B',
            'open_weight': True,
            'creator': 'Mistral',
            'version': 'instruct-v0.1'
        },
        'mixtral-8*7b': {
            'model_name': 'Mixtral-8*7B',
            'model_size': '8*7B',
            'open_weight': True,
            'creator': 'Mistral',
            'version': 'instruct-v0.1'
        },
        'glm-4-plus': {
            'model_name': 'GLM-4-Plus',
            'model_size': 'N/A',
            'open_weight': True,
            'creator': 'ZHIPU AI',
            'version': 'N/A'
        },
        'qwen2.5-72b': {
            'model_name': 'Qwen2.5-72B',
            'model_size': '72B',
            'open_weight': True,
            'creator': 'Qwen',
            'version': 'instruct'
        },
        'deepseek': {
            'model_name': 'Deepseek',
            'model_size': '236B',
            'open_weight': True,
            'creator': 'Deepseek',
            'version': 'v2.5'
        },
        'yi-lightning': {
            'model_name': 'Yi-Lightning',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': '01.ai',
            'version': 'N/A'
        },
        'qwen2-vl-72b': {
            'model_name': 'Qwen2-VL-72B',
            'model_size': '72B',
            'open_weight': True,
            'creator': 'Qwen',
            'version': 'instruct'
        },
        'glm-4v-plus': {
            'model_name': 'GLM-4V-Plus',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'ZHIPU AI',
            'version': 'N/A'
        },
        'llama-3.2-11b-v': {
            'model_name': 'Llama-3.2-11B-V',
            'model_size': '11B',
            'open_weight': True,
            'creator': 'Meta AI',
            'version': 'instruct'
        },
        'llama-3.2-90b-v': {
            'model_name': 'Llama-3.2-90B-V',
            'model_size': '90B',
            'open_weight': True,
            'creator': 'Meta AI',
            'version': 'instruct'
        },
        'dall-e 3': {
            'model_name': 'DALL-E 3',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'OpenAI',
            'version': 'N/A'
        },
        'sd-3.5': {
            'model_name': 'SD-3.5',
            'model_size': '8B',
            'open_weight': True,
            'creator': 'Stability AI',
            'version': 'large'
        },
        'flux-1.1': {
            'model_name': 'FLUX-1.1',
            'model_size': 'N/A',
            'open_weight': False,
            'creator': 'Black Forset Labs',
            'version': 'pro'
        },
        'playground 2.5': {
            'model_name': 'Playground 2.5',
            'model_size': 'N/A',
            'open_weight': True,
            'creator': 'Playground',
            'version': '1024px-aesthetic'
        },
        'hunyuan-dit': {
            'model_name': 'Hunyuan-DiT',
            'model_size': 'N/A',
            'open_weight': True,
            'creator': 'Tencent',
            'version': 'N/A'
        },
        'kolors': {
            'model_name': 'Kolors',
            'model_size': 'N/A',
            'open_weight': True,
            'creator': 'Kwai',
            'version': 'N/A'
        },
        'cogview-3-plus': {
            'model_name': 'CogView-3-Plus',
            'model_size': 'N/A',
            'open_weight': True,
            'creator': 'ZHIPU AI',
            'version': 'N/A'
        }
    }
    return model_info

def load_data(test_data_path, model_type):
    """
    Loads test data and leaderboard data based on model_type.
    """
    try:
        # Load test data
        test_data = pd.read_csv(test_data_path)
        # Format test data values to 4 decimal places
        value_column = test_data.columns[1]  # Get the second column name (value column)
        test_data[value_column] = test_data[value_column].round(4)
    except Exception as e:
        print(f"Error reading test data: {e}")
        exit(1)
    
    try:
        # Determine leaderboard path based on model_type
        leaderboard_path = os.path.join(os.path.dirname(__file__), 'data', f"{model_type}_leaderboard.csv")
        # Load leaderboard data
        leaderboard = pd.read_csv(leaderboard_path)
    except Exception as e:
        print(f"Error reading leaderboard data: {e}")
        exit(1)
    
    # Load model information
    model_info = load_model_info()

    # Get testing time with minute-level precision
    test_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')

    # Map model names to lower case
    test_data['model_lower'] = test_data['model'].str.lower()

    # Merge test data with model information
    model_info_df = pd.DataFrame.from_dict(model_info, orient='index').reset_index().rename(columns={'index': 'model_lower'})
    test_data = test_data.merge(model_info_df, on='model_lower', how='left')

    # Add testing time to each record
    test_data['test_time'] = test_time

    return test_data, leaderboard

def filter_leaderboard(leaderboard, aspect):
    """
    Filters the leaderboard to include only the specified aspect.
    """
    if (aspect not in leaderboard.columns):
        print(f"Aspect '{aspect}' not found in leaderboard data.")
        exit(1)
    
    # Keep only 'Model' and the specified aspect
    filtered_leaderboard = leaderboard[['Model', aspect]]
    
    return filtered_leaderboard

def generate_summary_with_gpt(test_data, leaderboard):
    """
    Generates a summary using OpenAI GPT-4 focused on analyzing test models' performance.
    """
    prompt = f"""
    Please analyze the performance of our test models compared to the leaderboard:
    
    Test Data (Our models):
    {test_data.to_string(index=False)}
    
    Leaderboard Reference:
    {leaderboard.to_string(index=False)}
    
    Note:
    - If a test model name matches a model in the leaderboard, it refers to this evaluation.
    - If there are duplicate model names, explain that the model's responses vary each time.
    - Emphasize that our dataset is robust, ensuring consistent performance for the same model across evaluations.
    
    Format your response strictly as follows:
    1. Start with one main heading "## Test Models Analysis"
    2. For each test model, create a sub-heading "### [Model Name]"
    3. Under each model section, analyze:
       - Its performance metrics
       - Comparison with state-of-the-art models
       - Strengths and areas for improvement
    4. End with "## Summary" section containing overall conclusions
    
    Use **bold** for important metrics and keep the analysis focused and concise.
    """
    
    messages=[
        {"role": "system", "content": "You are an AI model evaluator. Focus on analyzing the test models' performance compared to the leaderboard."},
        {"role": "user", "content": prompt}
    ]
    prompt = "\n".join([msg["content"] for msg in messages])
    response = service.process(prompt)

    return response

def generate_report(test_data, leaderboard, aspect, case_data, base_dir):
    """
    Generates the HTML report using Jinja2 templates.
    """
    # Load template
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    # Register the markdown filter
    env.filters['markdown'] = lambda text: markdown.markdown(text)
    template = env.get_template("report_template.html")
    
    # Prepare chart data
    chart_data = prepare_chart_data(leaderboard, test_data, aspect)
    
    # Generate summary using GPT-4
    summary = generate_summary_with_gpt(test_data, leaderboard)
    
    value_key = test_data.columns[1]  # Extract the dynamic key

    # Render HTML
    html_content = template.render(
        test_data=test_data.to_dict(orient="records"),
        leaderboard=leaderboard.to_dict(orient="records"),
        aspect=aspect,
        chart_data=chart_data,
        current_year=pd.Timestamp.now().year,
        summary=summary,
        download_link="/path/to/download",  # Update with actual download link
        attribute=value_key,  # Existing code
        case_data=case_data   # Pass the case data to the template
    )
    
    # Save HTML to base_dir
    OUTPUT_HTML = os.path.join(base_dir, "report.html")
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Copy static directory to base_dir/static
    src_static = os.path.join(TEMPLATE_DIR, "..", "static")
    dest_static = os.path.join(base_dir, "static")
    shutil.copytree(src_static, dest_static, dirs_exist_ok=True)

def prepare_chart_data(leaderboard, test_data, aspect):
    """
    Prepares data for Chart.js visualization with special handling for test models.
    """
    # Normalize model names for matching
    def normalize(name):
        return ''.join(e.lower() for e in name if e.isalnum())
    
    test_models = {normalize(model): model for model in test_data['model']}
    # Extract the key dynamically
    value_key = test_data.columns[1]
    test_data_dict = {normalize(row['model']): row[value_key] for _, row in test_data.iterrows()}
    
    # Add test models to the leaderboard with (new) suffix
    new_entries = []
    for normalized_model, original_model in test_models.items():
        # Always append (new) regardless of existence in leaderboard
        new_entries.append({'Model': f"{original_model} (new)", aspect: test_data_dict[normalized_model]})
    
    if new_entries:
        new_entries_df = pd.DataFrame(new_entries)
        leaderboard = pd.concat([leaderboard, new_entries_df], ignore_index=True)
    
    # Sort the combined leaderboard by the specified aspect
    leaderboard = leaderboard.sort_values(by=aspect, ascending=False)
    
    labels = leaderboard['Model'].tolist()
    data_values = leaderboard[aspect].tolist()
    
    # Prepare background colors
    background_colors = []
    for model in labels:
        if "(new)" in model:
            background_colors.append('rgba(255, 215, 0, 0.7)')  # Gold color
        else:
            background_colors.append('rgba(54, 162, 235, 0.5)')  # Default color
    
    chart_data = {
        'labels': labels,
        'datasets': [{
            'label': aspect,
            'data': data_values,
            'backgroundColor': background_colors,
            'borderColor': background_colors,  # Use the same color for borders
            'borderWidth': 1
        }]
    }
    return chart_data

def report_pipeline(base_dir, model_type, aspect):
    # Locate the unique CSV file in base_dir
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    if len(csv_files) != 1:
        print(f"Expected exactly one CSV file in {base_dir}, found {len(csv_files)}.")
        exit(1)
    test_data_path = csv_files[0]


    # Load data
    test_data, leaderboard = load_data(test_data_path, model_type)

    # Map test_data model names using return_dict
    test_data['model'] = test_data['model'].map(return_dict).fillna(test_data['model'])
    
    # Filter leaderboard
    filtered_leaderboard = filter_leaderboard(leaderboard, aspect)
    
    # Generate case data
    case_data = generate_case(
        base_dir=base_dir,
        model_type=model_type,
        test_model_list=test_data['model'].tolist(),
    )
    # Generate report
    generate_report(test_data, filtered_leaderboard, aspect, case_data, base_dir)
    
    print(f"Report generated successfully: {os.path.join(base_dir, 'report.html')}")

# Remove or modify the __main__ section
def find_unique_csv(base_dir):
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    if len(csv_files) != 1:
        raise ValueError(f"Expected exactly one CSV file in {base_dir}, found {len(csv_files)}.")
    return csv_files[0]

if __name__ == "__main__":
    base_dir = 'D:/Paper/TrustAGI-code/examples/advanced_ai_risk'  # Update to your base directory
    model_type = 'llm'
    aspect = 'Advanced.'
    generate_report(base_dir, model_type, aspect)
