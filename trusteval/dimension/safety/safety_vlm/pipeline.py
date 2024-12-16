from tqdm import tqdm
from .Aspect1_FigStep import main as figstep_main
from .Aspect2_JailbreakInPieces import main as JailbreakInPieces_main
from .Aspect3_MMSafetyBench import main as MMSafetyBench_main
from .Aspect4_VisualAdvEx import main as VisualAdvEx_main
from .Aspect5_VisualRolePlay import main as VisualRolePlay_main

def main(base_dir=None):
    tasks = [
        ("JailbreakInPieces_main", JailbreakInPieces_main),
        ("figstep_main", figstep_main),
        ("MMSafetyBench_main", MMSafetyBench_main),
        ("VisualRolePlay_main", VisualRolePlay_main),
        ("VisualAdvEx_main", VisualAdvEx_main)
    ]
    
    for task_name, task_func in tqdm(tasks, desc="Running tasks"):
        task_func(base_dir=base_dir, initialize=True)
        print(f"{task_name} completed")
