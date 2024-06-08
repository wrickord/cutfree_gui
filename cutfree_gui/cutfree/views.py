# System imports
import os
import sys
import logging
import subprocess
import asyncio

# Third-party imports
from django.shortcuts import render
from django.http import HttpResponse

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from models import CutFreeModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ENV = os.environ.copy()
ENV["JULIA_PROJECT"] = "JULIA_VENV"

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
    
    return result, degeneracy
    
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
    
    result, degeneracy = await run(
        starting_oligo, 
        blocking_sites, 
        min_blocks, 
        increase_diversity,
        use_rl
    )

    algorithm = "CutFreeRL" if use_rl else "CutFree"

    return result, degeneracy, algorithm

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

            cutfree_output, degeneracy, algorithm_used = await main(
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
                "algorithm_used": algorithm_used
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
                "algorithm_used": ""
            }
            
            return render(request, "cutfree.html", context)
    except Exception as e:
        logger.error(f"Error in index view: {e}")
        return HttpResponse("An error occurred: " + str(e))