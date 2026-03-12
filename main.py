import os

DATASETS = {
    "511285_Oakwood_Rd_Llanedeyrn": "511285.csv",
    "511320_Abercynon_St_Westmoors": "511320.csv",
    "511469_Heol_Briwnant_South_Llanishen": "511469.csv"
}

OUTPUT_DIR = "project_results"
CLEAN_DIR = os.path.join(OUTPUT_DIR, "cleaned_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)