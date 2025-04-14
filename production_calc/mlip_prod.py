import logging
import json
import os
from calc import get_calculator
from monty.serialization import dumpfn, loadfn
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import multiprocessing

# set up
mlip_method = "chgnet" # name of MLIP to run
input_file = "all_structures_no_energy.json" # dumpfn'd file that has {key: Structure} pairs
output_dir = Path(f"{mlip_method}/")
if not output_dir.exists():
    output_dir.mkdir()
calc = get_calculator(mlip_method)

# functions
def check_complete(structures: dict[str, Structure]):
    original_len = len(structures)
    jsonl_files = [f for f in os.listdir(output_dir)
                if f"{mlip_method}" in f and f.endswith(".jsonl")]
    for file in jsonl_files:
        file_path = os.path.join(output_dir, file)
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = data.get("key")
                    if key in structures.keys():
                        structures.pop(key)  # Remove processed keys
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}")
    logging.info(f"check_complete finished. original len {original_len}; len after checking: {len(structures)}")

def process_structure_dict(structures: dict[str, Structure], batch_idx):

    output_file_path = str(output_dir) + f"/{mlip_method}_output_energy_batch_{batch_idx}.jsonl"
    print(f"{output_file_path}")
    logging_counter = 0

    for key, structure in structures.items():

        # calculate energy
        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        ase_atoms.calc = calc
        energy = float(ase_atoms.get_potential_energy())

        # write output
        with open(output_file_path, "a") as f:
            json.dump({"key": key, "data": energy}, f)
            f.write("\n")
            f.flush()  # Ensure immediate writing to disk

        # logging
        logging_counter += 1
        if logging_counter >= 100:
            print(f"finished 100 calculations in batch {batch_idx}, most recent is {key}")
            logging_counter = 0

    print(f"finished batch {batch_idx}")

def production_run(nproc):

    structures = loadfn(f"{input_file}")
    logging.info("input file loaded")

    check_complete(structures)

    chunks = [{} for _ in range(nproc)]
    iproc = 0
    for key in list(structures.keys()):
        chunks[iproc][key] = structures[key]
        iproc = (iproc + 1) % nproc

    procs = []
    for iproc in range(nproc):
        proc = multiprocessing.Process(
            target=process_structure_dict,
            args=(chunks[iproc], iproc),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, force=True)

    production_run(nproc=2)
