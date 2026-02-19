import streamlit as st
import muspy, glob, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from midi2audio import FluidSynth
import pretty_midi
import soundfile as sf
from google import genai
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, midi_files, stoi, seq_len=50):
        self.data = []
        self.seq_len = seq_len
        print

        for path in midi_files:
            events = midi_to_events(path)
            idxs = [stoi[e] for e in events if e in stoi]
            print(f"File: {path}, Events: {len(events)}, Valid idxs: {len(idxs)}")  # Debug
            for i in range(max(0,(len(idxs) - seq_len))):
                x = idxs[i:i+seq_len]
                y = idxs[i+1:i+seq_len+1]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return torch.tensor(x), torch.tensor(y)


def pitches_in_bar(track, bar_start, bar_end):
    pcs = []
    for n in track.notes:
        if n.time >= bar_start and n.time < bar_end:
            pcs.append(n.pitch % 12)
    return pcs

def detect_chord_from_pcs(pcs):
    if not pcs:
        return "N"
    counts = np.bincount(pcs, minlength=12)
    root = int(np.argmax(counts))
    has_major3 = ((root + 4) % 12) in pcs
    has_minor3 = ((root + 3) % 12) in pcs
    if has_major3:
        quality = "maj"
    elif has_minor3:
        quality = "min"
    else:
        quality = "other"
    return f"{root}:{quality}"


def build_vocab():
    vocab = []
    vocab += [f"NOTE_ON_{i}" for i in range(128)]
    vocab += [f"NOTE_OFF_{i}" for i in range(128)]
    vocab += [f"TIME_SHIFT_{i}" for i in range(1, 25)]  # 24 = 1 beat if 24 PPQ
    vocab += ["END"]

    stoi = {t:i for i,t in enumerate(vocab)}
    itos = {i:t for t,i in stoi.items()}
    return stoi, itos

def train_model(model, loader, epochs=20):
    model.train()
    print("Training started")

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

def midi_to_events(path, ppq=24):
    pm = pretty_midi.PrettyMIDI(path)
    events = []

    all_notes = []
    for inst in pm.instruments:
        for note in inst.notes:
            all_notes.append(("on", note.start, note.pitch))
            all_notes.append(("off", note.end, note.pitch))

    all_notes.sort(key=lambda x: x[1])

    current_time = 0.0
    for typ, t, pitch in all_notes:
        delta = t - current_time
        steps = int(round(delta * ppq))
        while steps > 0:
            shift = min(steps, 24)
            events.append(f"TIME_SHIFT_{shift}")
            steps -= shift
        current_time = t

        if typ == "on":
            events.append(f"NOTE_ON_{pitch}")
        else:
            events.append(f"NOTE_OFF_{pitch}")

    events.append("END")
    return events

class PolyphonicRNN(nn.Module):
    def __init__(self, vocab_size, emb=256, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.lstm(x, h)
        return self.fc(y), h


midi_files = glob.glob("./testdata2/*.mid", recursive=True)
        
stoi, itos = build_vocab()
dataset = MusicDataset(midi_files, stoi)
print(len(dataset))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PolyphonicRNN(len(stoi)).to(device)
model.load_state_dict(torch.load("polyphonic_rnn.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
loader = DataLoader(dataset, batch_size=1, shuffle=True)
train_model(model, loader, epochs=5)
torch.save(model.state_dict(), "polyphonic_rnn.pt")
print("Trained!")
