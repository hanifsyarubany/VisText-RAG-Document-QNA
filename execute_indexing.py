import requests
import argparse

def args():
    parser = argparse.ArgumentParser(description="Initialization flag parser")
    parser.add_argument("--initialize", 
                        action="store_true",  # sets to True if --initialize is provided
                        help="Flag to perform initialization")
    return parser.parse_args()

def main():
    print("START INDEXING ...\n")
    arg = args()
    url = "http://127.0.0.1:8000/indexing"
    if arg.initialize:
        result = requests.post(url, json={"initialize": True}).json()
        print(result["response"])
    else:
        result = requests.post(url, json={"initialize": False}).json()
        print(result["response"])
if __name__ == "__main__":
    main()