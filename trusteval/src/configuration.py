import yaml
import os
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

console = Console()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/config.yaml")

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

def save_config(config):
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


SUPPORTED_MODELS = {
    # Note: We will continuously update supported models beyond those listed below.
    "OPENAI": [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-3.5-turbo',
        'text-embedding-ada-002',
        'dalle3'
    ],
    "AZURE": [
        'gpt-4o',
        'gpt-4o-mini', 
        'gpt-3.5-turbo',
        'text-embedding-ada-002',
        'dalle3'
    ],
    "DEEPINFRA": [
        'qwen-2.5-72B',
        'gemma-2-27B',
        'llama-2-13B',
        'llama-3.2-11B-V',
        'llama-3.2-90B-V',
        'llama-3-70B',
        'llama-3-8B',
        'llama-3.1-70B',
        'llama-3.1-8B',
        'mistral-7B',
        'mistral-8x22B',
        'mistral-8x7B'
    ],
    "ANTHROPIC": [
        'claude-3.5-sonnet',
        'claude-3-haiku'
    ],
    "GOOGLE": [
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ],
    "COHERE": [
        'command-r',
        'command-r-plus'
    ],
    "ZHIPU": [
        'glm-4',
        'glm-4-plus',
        'glm-4v',
        'glm-4v-plus',
        'cogview-3-plus'
    ],
    "DEEPSEEK": [
        'deepseek-chat'
    ],
    "REPLICATE": [
        'flux-1.1-pro',
        'playground-v2.5'
    ],
    "YI": [
        'yi-lightning'
    ],
    "QWEN": [
        'qwen-vl-max-0809'
    ],
    "INTERN": [
        'internLM-72B'
    ],
    "OPENROUTER": [
        'qwen-vl-max-0809'
    ],
    "DISCOVERY": [
        # No models mapped in the provided config.yaml
    ],
    "LOCAL": [
        'HunyuanDiT',
        'kolors',
        'playground-v2.5',
        'sd-3.5-large',
        'sd-3.5-large-turbo'
    ]
}

def display_available_services(config):
    table = Table(title="Available Services")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Supported Models", style="magenta")
    
    services = {
        "OPENAI": ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
        "AZURE": ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_API_VERSION"],
        "DEEPINFRA": ["DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"],
        "ANTHROPIC": ["ANTHROPIC_API_KEY"],
        "GOOGLE": ["GOOGLE_API_KEY"],
        "COHERE": ["COHERE_API_KEY"],
        "ZHIPU": ["ZHIPU_API_KEY", "ZHIPU_BASE_URL"],
        "DEEPSEEK": ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"],
        "REPLICATE": ["REPLICATE_API_TOKEN"],
        "YI": ["YI_API_KEY", "YI_BASE_URL"],
        "QWEN": ["QWEN_API_KEY", "QWEN_BASE_URL"],
        "INTERN": ["INTERN_API_KEY", "INTERN_BASE_URL"],
        "OPENROUTER": ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"],
        "DISCOVERY": ["DISCOVERY_API_KEY", "DISCOVERY_BASE_URL"],
        "LOCAL": ["LOCAL_API_KEY", "LOCAL_BASE_URL"]
    }
    
    for service, keys in services.items():
        exists = all(key in config.get(service, {}) for key in keys)
        status = "[bold green]Configured[/bold green]" if exists else "[bold red]Incomplete[/bold red]"
        models = ", ".join(SUPPORTED_MODELS.get(service, ["N/A"]))
        table.add_row(service, status, models)
    
    # Handle BING_SEARCH separately
    bing_keys = ["BING_SEARCH_V7_SUBSCRIPTION_KEY", "BING_SEARCH_V7_ENDPOINT"]
    bing_exists = all(key in config for key in bing_keys) and all(config[key] for key in bing_keys)
    bing_status = "[bold green]Configured[/bold green]" if bing_exists else "[bold red]Incomplete[/bold red]"
    bing_models = "N/A"  # Update if BING_SEARCH supports models
    table.add_row("BING_SEARCH", bing_status, bing_models)
    
    console.print(table)

def configure_service(config, service_name, required_keys):
    console.print(f"\n[bold cyan]Configuring {service_name}[/bold cyan]", style="bold magenta")
    
    service_config = config.get(service_name, {})
    for key in required_keys:
        exists = key in service_config
        status = "[bold green]Exists[/bold green]" if exists else "[bold red]Missing[/bold red]"
        console.print(f"{key}: {status}")
        
        new_value = Prompt.ask(
            f"Enter {key} (Press Enter to skip)",
            password=True if "KEY" in key or "TOKEN" in key else False
        )
        
        if new_value.strip():
            service_config[key] = new_value
            console.print(f"[bold green]✓[/bold green] {key} updated successfully!", style="bold green")
        else:
            console.print(f"[bold yellow]→[/bold yellow] Skipped modifying {key}.", style="bold yellow")
    
    config[service_name] = service_config
    return config

def configuration():
    console.print(Panel.fit(
        "[bold green]Welcome to TrustEval Config Manager[/bold green]\n"
        "Configure your API keys for various services seamlessly.",
        style="bold blue"
    ))
    console.print("[bold yellow]Note: The list of supported models will be continuously updated.[/bold yellow]")
    
    config = load_config()
    
    while True:
        display_available_services(config)
        
        service = Prompt.ask(
            "\nWhich service would you like to configure?",
            choices=[
                "OPENAI", "AZURE", "DEEPINFRA", "ANTHROPIC", "GOOGLE",
                "COHERE", "ZHIPU", "DEEPSEEK", "REPLICATE", "YI", "QWEN",
                "INTERN", "OPENROUTER", "DISCOVERY", "LOCAL",
                "configure BING_SEARCH", "exit"
            ],
            default="exit"
        )
        
        if service.lower() == "exit":
            break
        
        if service == "BING_SEARCH":
            bing_sub_key = Prompt.ask(
                "Enter BING_SEARCH_V7_SUBSCRIPTION_KEY (Press Enter to skip)",
                password=True
            )
            if bing_sub_key.strip():
                config["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = bing_sub_key
                console.print(f"[bold green]✓[/bold green] BING_SEARCH_V7_SUBSCRIPTION_KEY updated successfully!", style="bold green")
            
            bing_endpoint = Prompt.ask(
                "Enter BING_SEARCH_V7_ENDPOINT (Press Enter to skip)"
            )
            if bing_endpoint.strip():
                config["BING_SEARCH_V7_ENDPOINT"] = bing_endpoint
                console.print(f"[bold green]✓[/bold green] BING_SEARCH_V7_ENDPOINT updated successfully!", style="bold green")
        else:
            services_keys = {
                "OPENAI": ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                "AZURE": ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_API_VERSION"],
                "DEEPINFRA": ["DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"],
                "ANTHROPIC": ["ANTHROPIC_API_KEY"],
                "GOOGLE": ["GOOGLE_API_KEY"],
                "COHERE": ["COHERE_API_KEY"],
                "ZHIPU": ["ZHIPU_API_KEY", "ZHIPU_BASE_URL"],
                "DEEPSEEK": ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"],
                "REPLICATE": ["REPLICATE_API_TOKEN"],
                "YI": ["YI_API_KEY", "YI_BASE_URL"],
                "QWEN": ["QWEN_API_KEY", "QWEN_BASE_URL"],
                "INTERN": ["INTERN_API_KEY", "INTERN_BASE_URL"],
                "OPENROUTER": ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"],
                "DISCOVERY": ["DISCOVERY_API_KEY", "DISCOVERY_BASE_URL"],
                "LOCAL": ["LOCAL_API_KEY", "LOCAL_BASE_URL"]
            }
            config = configure_service(config, service, services_keys[service])
        
        save_config(config)
        
        if service != "configure BING_SEARCH":
            console.print(f"\n[green]✓[/green] {service} configuration saved successfully!", style="bold green")
        
        if not Confirm.ask("\nWould you like to configure another service?"):
            break
    
    console.print("\n[bold green]Configuration complete! Your config.yaml has been updated.[/bold green]", style="bold yellow")
    
if __name__ == "__main__":
    configuration()