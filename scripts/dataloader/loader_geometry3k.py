import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData


MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}


def generate_data(data_path: str):
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        image = Image.open(os.path.join(folder_path, "img_diagram.png"), "r")
        with open(os.path.join(folder_path, "data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            yield {
                "images": [image],
                "problem": "<image>" + data["annotat_text"],
                "answer": data["choices"][MAPPING[data["answer"]]],
                "id": data["id"],
                "choices": data["choices"],
                "ground_truth": data["answer"],
            }


def main():
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "train")})
    valset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "val")})
    testset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "test")})
    dataset = DatasetDict({"train": trainset, "validation": valset, "test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub("hiyouga/geometry3k")


if __name__ == "__main__":
    main()
