# System imports
import os
import sys
import logging
import asyncio

# Third-party imports
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from models import CutFreeModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
ENV = os.environ.copy()
ENV["JULIA_PROJECT"] = "JULIA_VENV"
IUB_CODES = {
    "A": np.array(["A"]),
    "C": np.array(["C"]),
    "G": np.array(["G"]),
    "T": np.array(["T"]),
    "R": np.array(["A", "G"]),
    "Y": np.array(["C", "T"]),
    "S": np.array(["C", "G"]),
    "W": np.array(["A", "T"]),
    "K": np.array(["G", "T"]),
    "M": np.array(["A", "C"]),
    "B": np.array(["C", "G", "T"]),
    "D": np.array(["A", "G", "T"]),
    "H": np.array(["A", "C", "T"]),
    "V": np.array(["A", "C", "G"]),
    "N": np.array(["A", "C", "G", "T"])
}

# Helper functions to call algorithms
def select_algorithm(starting_oligo, restriction_sites):
    model_info = [1, 1] # Version, model number
    
    # Predict if rbs is present
    model = CutFreeModel(
        load_model=True, 
        model_info=model_info 
    )
    preds, _ = model.predict([restriction_sites], [starting_oligo])

    if preds[0] == 0:
        return "cutfree"
    else:
        return "cutfree_rl"

def check_randomer(randomer, recognition_sites):
    for recognition_site in recognition_sites:
        for index in range(len(randomer) - len(recognition_site) + 1):
            counter = 0
            window = randomer[index : index + len(recognition_site)]

            # Check each recognition site base
            for char, base in zip(window, recognition_site):
                char = char.upper()
                base = base.upper()

                # Skip if either is a dash
                if char == "-" or base == "-":
                    counter += 1
                    continue

                if np.any(np.isin(IUB_CODES[char], IUB_CODES[base])):
                    counter += 1

            # Check to see if the entire recognition site is in the window
            if counter == len(recognition_site):
                return False

    return True

async def run(starting_oligo, 
              blocking_sites, 
              min_blocks, 
              increase_diversity,
              use_rl):
    increase_diversity = "true" if increase_diversity == "yes" else "false"
    if use_rl:
        arguments = f'cutfree_rl("{starting_oligo}", "{blocking_sites}")'
        command = [
            'julia', 
            '-e', 
            f'include("algorithms/cutfree_rl.jl"); {arguments}'
        ]
        process = await asyncio.create_subprocess_exec(
            *command, 
            stdout=asyncio.subprocess.PIPE,
            env=ENV
        )
        stdout, _ = await process.communicate()
        result = stdout.decode().strip()
    else:
        arguments = f'cutfree("{starting_oligo}", "{blocking_sites}",' + \
            f'{min_blocks}, "{increase_diversity}")'
        command = [
            'julia', 
            '-e', 
            f'include("algorithms/cutfree.jl"); {arguments}'
        ]
        process = await asyncio.create_subprocess_exec(
            *command, 
            stdout=asyncio.subprocess.PIPE,
            env=ENV
        )
        stdout, _ = await process.communicate()
        result = stdout.decode().strip().split("\n")[-1]
    
    if "-" in result:
        algo_used = "CutFree" if use_rl else "CutFreeRL"
        result = "Error: Algorithm used could not find a solution. Please " + \
            f"try again with the {algo_used} algorithm."
        degeneracy = 0
    elif " " in result.strip():
        result = "Error: Invalid input. Please check your input and try again."
        degeneracy = 0
    else: 
        arguments = f'get_degeneracy("{result}")'
        command = [
            'julia', 
            '-e', 
            f'include("algorithms/cutfree.jl"); {arguments}'
        ]
        process = await asyncio.create_subprocess_exec(
            *command, 
            stdout=asyncio.subprocess.PIPE,
            env=ENV
        )
        stdout, _ = await process.communicate()
        degeneracy = stdout.decode().strip()

        check_result = check_randomer(
            result, 
            blocking_sites.split(", ")
        )
    
    return result, degeneracy, check_result
    
async def main(starting_oligo, 
               blocking_sites,
               min_blocks, 
               increase_diversity, 
               algorithm_choice):
    
    if min_blocks > 1 \
        or increase_diversity == "yes" \
        or algorithm_choice == "cutfree":
        use_rl = False
    elif algorithm_choice == "cutfree_rl":
        use_rl = True
    elif algorithm_choice == "auto":
        use_rl = True if select_algorithm(
            blocking_sites,
            starting_oligo 
        ) == "cutfree_rl" else False
    else:
        return "Error: Invalid algorithm choice or parameters.", 0, "NONE"
    
    result, degeneracy, check_result = await run(
        starting_oligo, 
        blocking_sites, 
        min_blocks, 
        increase_diversity,
        use_rl
    )

    algorithm = "CutFreeRL" if use_rl else "CutFree"

    return result, degeneracy, algorithm, check_result

# View for the cutfree page
async def index(request):
    try:
        if request.method == "POST" and "starting_oligo" in request.POST:
            starting_oligo = str(request.POST.get("starting_oligo"))
            raw_blocking_sites = str(request.POST.getlist('blocking_sites'))
            blocking_sites = raw_blocking_sites.strip("[]").strip("'")
            min_blocks = int(request.POST.get("min_blocks"))
            increase_diversity = str(request.POST.get("increase_diversity"))
            algorithm_choice = str(request.POST.get("algorithm_choice"))

            cutfree_output, degeneracy, algorithm_used, check_result = \
                await main(
                    starting_oligo, 
                    blocking_sites, 
                    min_blocks, 
                    increase_diversity, 
                    algorithm_choice
                )
            
            context = {
                "starting_oligo": starting_oligo,
                "blocking_sites": blocking_sites,
                "min_blocks": min_blocks,
                "increase_diversity": increase_diversity,
                "algorithm_choice": algorithm_choice,
                "cutfree_output_value": cutfree_output, 
                "degeneracy_value": degeneracy,
                "algorithm_used": algorithm_used,
                "verified": check_result
            }
            
            return render(request, "cutfree.html", context)
        else:
            context = {
                "starting_oligo": "",
                "blocking_sites": "",
                "min_blocks": 1,
                "increase_diversity": "",
                "algorithm_choice": "",
                "cutfree_output_value": "", 
                "degeneracy_value": "",
                "algorithm_used": "",
                "verified": None
            }
            
            return render(request, "cutfree.html", context)
    except Exception as e:
        logger.error(f"Error in index view: {e}")
        return HttpResponse("An error occurred: " + str(e))