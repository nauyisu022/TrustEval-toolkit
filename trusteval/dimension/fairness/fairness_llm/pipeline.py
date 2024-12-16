import os
from src.utils import colored_print as print
from .preference import PreferenceGenerator
from .disparagement import DisparagementGenerator
from .preference import PreferenceGenerator



def main(base_dir=None, ):
    print("Running StereotypeGenerator ...")
    
    # stereotype_generator=StereotypeGenerator(base_dir)
    # stereotype_generator.run()
    
    print("Running PreferenceGenerator ...")
    preference_generator=PreferenceGenerator(base_dir)
    preference_generator.run()
    
    
    print("Running DisparagementGenerator ...")
    disparagement_generator=DisparagementGenerator(base_dir)
    disparagement_generator.run()
    print("All dataset generation finished.",color="GREEN")

    
        
