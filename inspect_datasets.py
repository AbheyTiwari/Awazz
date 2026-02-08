from datasets import load_dataset

ds1 = load_dataset("Hemg/Emotion-audio-Dataset", split="train")

print(ds1.features)