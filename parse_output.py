from monty.serialization import loadfn, dumpfn
import json
import os

def parse_jsonl_output(method, output_dir=None):
    if not output_dir:
        output_dir = method

    jsonl_files = [f for f in os.listdir(output_dir) if f"{method}" in f and f.endswith(".jsonl")]
    parsed_dict = {}

    for file in jsonl_files:
        file_path = os.path.join(output_dir, file)
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = data.get("key")
                    value = data.get("data")
                    parsed_dict[key] = value
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}")

    print(method, len(parsed_dict))
    return parsed_dict

def jsonl_dict_to_res_dict(parsed_dict):
    res_dict = {}

    for key, energy in parsed_dict.items():
        id, hop_key, image_idx = key.split(".")

        if not id in res_dict.keys():
            res_dict[id] = {}
        if not hop_key in res_dict[id].keys():
            res_dict[id][hop_key] = []

        res_dict[id][hop_key].insert(int(image_idx), energy)

    print(len(res_dict))
    return res_dict

def parse_all_output(methods, dump=False):
    all_res = {}

    for method in methods:
        jsonl_dict = parse_jsonl_output(method)
        res_dict = jsonl_dict_to_res_dict(jsonl_dict)
        all_res[method] = res_dict

    if dump:
        dumpfn(all_res, "all_calc_res.json")

    return all_res


if __name__ == "__main__":
    methods=["DFT", "MACE", "GRACE", "mattersim", "deepmd", "eqV2-M", "sevenn", "CHGNet"]
    all_res = parse_all_output(
        methods=methods,
        dump=True
    )