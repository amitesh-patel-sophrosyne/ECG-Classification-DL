import os
import numpy as np
import wfdb
from scipy.signal import resample
import re


class Preprocess:
    def __init__(self, fs=250, target_fs=128):
        self.fs = fs
        self.target_fs = target_fs

    def downsample(self, signal):
        if self.fs != self.target_fs:
            num_samples = int(signal.shape[0] * self.target_fs / self.fs)
            return resample(signal, num_samples, axis=0)
        return signal

    def z_score_normalize(self, signal):
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0)
        std[std == 0] = 1
        return (signal - mean) / std
    
    def fix_polarity(self, signal):
        """
        Heuristic: if Channel 0 has negative mean, it's likely inverted.
        Flip it to ensure positive R-wave polarity.
        """
        if np.mean(signal[:, 0]) < 0:
            signal[:, 0] = -signal[:, 0]
        return signal

    def process(self, signal):
        # signal = self.fix_polarity(signal) 
        signal = self.downsample(signal)
        signal = self.z_score_normalize(signal)
        return signal


class SegmentExtractor:
    def __init__(self, segment_duration_sec=10, target_fs=128):
        self.segment_samples = segment_duration_sec * target_fs

    def convert_intervals_afdb(self, afib_intervals, ratio):
        return [(int(start * ratio), int(end * ratio)) for start, end in afib_intervals]
    
    def convert_intervals_ltafdb(self, labeled_intervals, ratio):
        return [(int(start * ratio), int(end * ratio), label) for start, end, label in labeled_intervals]

    def extract_segments_afdb(self, signal, afib_intervals, target_fs):
        segments = []
        labels = []

        total_samples = signal.shape[0]
        current = 0
        all_intervals = []

        for afib_start, afib_end in afib_intervals:
            if current < afib_start:
                all_intervals.append((current, afib_start, 0))  # non-AFib
            all_intervals.append((afib_start, afib_end, 1))  # AFib
            current = afib_end

        if current < total_samples:
            all_intervals.append((current, total_samples, 0))  # trailing non-AFib

        for start, end, label in all_intervals:
            interval_len = end - start
            num_full_segments = interval_len // self.segment_samples

            for i in range(num_full_segments):
                seg_start = start + i * self.segment_samples
                seg_end = seg_start + self.segment_samples
                segment = signal[seg_start:seg_end].T  # shape: (2, 1280)
                segments.append(segment)
                labels.append(label)

        return segments, labels
    
    def extract_segments_ltafdb(self, signal, labeled_intervals, target_fs):
        segments = []
        labels = []

        for start, end, label in labeled_intervals:
            interval_len = end - start
            num_full_segments = interval_len // self.segment_samples

            for i in range(num_full_segments):
                seg_start = start + i * self.segment_samples
                seg_end = seg_start + self.segment_samples
                segment = signal[seg_start:seg_end].T  # shape: (2, segment_samples)
                segments.append(segment)
                labels.append(label)

        return segments, labels


def parse_afib_intervals(annotation, signal_length):
    """
    Parses AFIB intervals from the annotation for the AFDB dataset.
    They mentioned in the paper that they only take AFIB intervals, and rest of the intervals are Non-AF."""
    samples = annotation.sample
    aux_notes = annotation.aux_note

    afib_intervals = []
    afib_start = None

    for i, note in enumerate(aux_notes):
        if note == '(AFIB':
            afib_start = samples[i]
        elif note.startswith('(') and afib_start is not None:
            afib_end = samples[i]
            afib_intervals.append((afib_start, afib_end))
            afib_start = None

    if afib_start is not None:
        afib_intervals.append((afib_start, signal_length))

    return afib_intervals

def parse_afib_and_normal_intervals(annotation, signal_length, data_used):
    """
    Parses AFIB and Normal intervals from the annotation for the LTAFDB dataset.
    As mentioned in the paper, to take only AFIB and Normal intervals,
    """
    def clean_note(note):
        return re.sub(r'[\x00-\x1F\x7F]', '', note)
    
    samples = annotation.sample
    aux_notes = annotation.aux_note

    intervals = []
    current_start = None
    current_label = None

    for i, note in enumerate(aux_notes):
        if data_used != 'ltafdb':
            note = clean_note(note)  # Clean the note from any control characters

        if note == '(AFIB':
            if current_start is not None:
                intervals.append((current_start, samples[i], current_label))
            current_start = samples[i]
            current_label = 1
        elif note == '(N':
            if current_start is not None:
                intervals.append((current_start, samples[i], current_label))
            current_start = samples[i]
            current_label = 0
        elif note.startswith('(') and current_start is not None:
            # Any new rhythm starting -> end current interval
            intervals.append((current_start, samples[i], current_label))
            current_start = None
            current_label = None

    # If signal ends with an open interval
    if current_start is not None:
        intervals.append((current_start, signal_length, current_label))

    # Return only AFIB and Normal intervals
    return [(start, end, label) for start, end, label in intervals if label in (0, 1)]



def create_dataset_from_paths(record_paths, preprocess, data_used='afdb'):
    extractor = SegmentExtractor(segment_duration_sec=10, target_fs=preprocess.target_fs)
    all_segments = []
    all_labels = []

    for record_base in record_paths:
        dat_path = record_base + '.dat'
        atr_path = record_base + '.atr'
        if not os.path.exists(dat_path) or not os.path.exists(atr_path):
            print(f"Skipping {record_base}, missing .dat or .atr file.")
            continue

        record = wfdb.rdrecord(record_base)
        annotation = wfdb.rdann(record_base, 'atr')

        signal = record.p_signal[:, :2]  # Take both channels
        if data_used == 'afdb':
            # For AFDB, parse AFIB intervals
            print(f"Processing {data_used} record: {record_base}")
            afib_intervals = parse_afib_intervals(annotation, len(signal))
        elif (data_used == 'ltafdb' or data_used == 'mitdb'):
            # For LTAFDB, parse AFIB and Normal intervals
            print(f"Processing {data_used} record: {record_base}")
            afib_intervals = parse_afib_and_normal_intervals(annotation, len(signal), data_used)
        else:
            raise ValueError(f"Unknown dataset type: {data_used}")
        
        # Preprocess signal first (downsample and normalize)
        processed_signal = preprocess.process(signal)

        # Convert AFib intervals to new sample scale
        ratio = preprocess.target_fs / preprocess.fs
        if data_used == 'afdb':
            afib_intervals_ds = extractor.convert_intervals_afdb(afib_intervals, ratio)
        elif (data_used == 'ltafdb' or data_used == 'mitdb'):
            afib_intervals_ds = extractor.convert_intervals_ltafdb(afib_intervals, ratio)
        else:
            raise ValueError(f"Unknown dataset type: {data_used}")

        if data_used == 'afdb':
            segments, labels = extractor.extract_segments_afdb(
                processed_signal, afib_intervals_ds, preprocess.target_fs
            )
        elif (data_used == 'ltafdb' or data_used == 'mitdb'):
            segments, labels = extractor.extract_segments_ltafdb(
                processed_signal, afib_intervals_ds, preprocess.target_fs
            )
        else:
            raise ValueError(f"Unknown dataset type: {data_used}")

        all_segments.extend(segments)
        all_labels.extend(labels)

    return np.array(all_segments), np.array(all_labels)


if __name__ == "__main__":
    output_dir = 'data/afdb_data'
    preprocessor = Preprocess(fs=250, target_fs=128)

    X, y = create_dataset_from_paths(output_dir, preprocess=preprocessor)

    print("X shape:", X.shape)  # (N, 2, 1280)
    print("y shape:", y.shape)  # (N,)
    print("AFib segments:", np.sum(y))
    print("Non-AFib segments:", len(y) - np.sum(y))
