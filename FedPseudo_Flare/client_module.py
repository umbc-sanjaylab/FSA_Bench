import sys 
import runpy  

if __name__ == "__main__": 
    project_root = "/home/il66977@ad.umbc.edu/Benchmarking_FSA"
    flare_root = "/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare"
        
    sys.path.insert(0, project_root)
    sys.path.insert(0, flare_root)
    runpy.run_module("FedPseudo_Flare.client", run_name="__main__")  