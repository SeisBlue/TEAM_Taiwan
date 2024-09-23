import json
import torch

from model.ttsam_model import get_full_model


def get_sample(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    waveform = torch.tensor(data["waveform"]).to(torch.double).unsqueeze(0)
    input_station = torch.tensor(data["sta"]).to(torch.double).unsqueeze(0)
    target_station = torch.tensor(data["target"]).to(torch.double).unsqueeze(0)
    sample = {"waveform": waveform, "sta": input_station,
              "target": target_station}
    return sample


if __name__ == "__main__":
    model_path = f"model/ttsam_trained_model_11.pt"
    full_Model = get_full_model(model_path)

    sample = get_sample("tests/data/ttsam_sample.json")
    weight, sigma, mu = full_Model(sample)

    print(f"weight: {weight}")
    print(f"sigma: {sigma}")
    print(f"mu: {mu}")
