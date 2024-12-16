import os,sys
from . import stereotype
from . import  preference

def run(base_dir=None):
    print("Running StereotypeGenerator ...")
    stereotype.main(base_dir)
    
    print("Running PreferenceGenerator ...")
    preference.main(base_dir,20)
    

    
    
        
