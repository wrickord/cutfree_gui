# System imports
import os
import sys

# Third-party imports
from django.shortcuts import render
from julia.api import Julia

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from models import CutFreeModel

# Initialize Julia upon server start
jl = Julia(compiled_modules=False)
jl.eval('import Pkg')
jl.eval('Pkg.activate(\"JULIA_VENV\")')

UNRECOGNIZED_PACKAGES = True
if UNRECOGNIZED_PACKAGES:
    pkgs = [
        "ArgParse", "BioSequences", "CSV", "Clp", "Conda", "DataFrames", 
        "DataStructures", "GLPK", "Gurobi", "JuMP", "NamedArrays", "PyCall",
        "Random", "StatsBase", "Suppressor", "XLSX"
    ]
    for pkg in pkgs:
        jl.eval(f'Pkg.add(\"{pkg}\")')
        jl.eval(f'Pkg.update(\"{pkg}\")')
jl.eval('Pkg.instantiate()')
jl.eval('using ArgParse, PyCall, Conda')
jl.eval('Conda.add(\"scikit-learn\")')
jl.eval(f'include(\"algorithms/cutfree.jl\")')
jl.eval(f'include(\"algorithms/cutfree_rl.jl\")')

# Helper functions to call algorithms
def select_algorithm(starting_oligo, restriction_sites):
    M = CutFreeModel()
    result = M.predict(
        starting_oligo,
        restriction_sites
    )

    return result

def run(starting_oligo, 
        blocking_sites_str, 
        min_blocks, 
        increase_diversity,
        use_rl):
    if increase_diversity == "yes":
        increase_diversity = "true"
    else:
        increase_diversity = "false"

    if use_rl:
        algorithm = "cutfree_rl"
        result = str(
            jl.eval(f'cutfree_rl("{starting_oligo}", [{blocking_sites_str}])')
        )
    else:
        algorithm = "cutfree"
        result = jl.eval(
            f'cutfree("{starting_oligo}", \
            [{blocking_sites_str}], \
            {min_blocks}, \
            \"{increase_diversity}\")'
        )
    degeneracy = str(jl.eval(f'get_degeneracy("{result}")'))

    return result, degeneracy, algorithm
    
def main(starting_oligo, 
         blocking_sites_list, 
         blocking_sites_str, 
         min_blocks, 
         increase_diversity, 
         algorithm_choice):
    
    if min_blocks > 1 \
        or increase_diversity == "yes" \
        or algorithm_choice == "cutfree":
        use_rl = False
    elif algorithm_choice == "cutfree_rl":
        use_rl = True
    elif algorithm_choice == "cutfree_rl":
        use_rl = select_algorithm(starting_oligo, blocking_sites_list)
    else:
        return "Error: Invalid algorithm choice or parameters.", 0, "NONE"
    
    result, degeneracy, algorithm = run(
        starting_oligo, 
        blocking_sites_str, 
        min_blocks, 
        increase_diversity,
        use_rl
    )

    return result, degeneracy, algorithm

# View for the home page
def home(request):
    if request.method == "POST":
        starting_oligo = str(request.POST.get('starting_oligo'))
        raw_blocking_sites = str(request.POST.getlist('blocking_sites'))
        blocking_sites_list = [
            s.strip().strip("'") for s in raw_blocking_sites[1:-1].split(',')
        ]
        blocking_sites_str = ", ".join([f'"{s}"' for s in blocking_sites_list])
        min_blocks = int(request.POST.get('min_blocks'))
        increase_diversity = str(request.POST.get('increase_diversity'))
        algorithm_choice = str(request.POST.get('algorithm_choice'))

        cutfree_output, degeneracy, algorithm_used = main(
            starting_oligo, 
            blocking_sites_list, 
            blocking_sites_str, 
            min_blocks, 
            increase_diversity, 
            algorithm_choice
        )
        
        context = {
            'cutfree_output_value': cutfree_output, 
            'degeneracy_value': degeneracy, 
            'algorithm_choice': algorithm_used
        }
        
        return render(request, "cutfree.html", context)
    else:
        context = {
            'cutfree_output_value': "NONE", 
            'degeneracy_value': 0, 
            'algorithm_choice': "NONE"
        }
        
        return render(request, "cutfree.html", context)
    

# LOCAL_TEST = False
# if LOCAL_TEST:
#     starting_oligo = "NNNNNNNNNNNNNNNNNNNNNNN"
#     raw_blocking_sites = "['GGGTCCC, GGCCTT']"
#     blocking_sites_list = [
#         s.strip().strip("'") for s in raw_blocking_sites[1:-1].split(',')
#     ]
#     blocking_sites_str = ", ".join([f'"{s}"' for s in blocking_sites_list])
#     min_blocks = int("1")
#     increase_diversity = str("yes")
#     algorithm_choice = str("auto")

#     result, degeneracy, algorithm = main(
#         starting_oligo, 
#         blocking_sites_list, 
#         blocking_sites_str, 
#         min_blocks, 
#         increase_diversity, 
#         algorithm_choice
#     )

#     print(result)