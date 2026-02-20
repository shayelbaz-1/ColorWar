import wave
import struct
import math
import os

SAMPLE_RATE = 44100
AMPLITUDE = 8000.0  # max 32767

def build_square_wave(freq, duration_sec, amplitude=AMPLITUDE):
    num_samples = int(SAMPLE_RATE * duration_sec)
    samples = []
    
    if freq == 0:
        return [0] * num_samples
        
    period = SAMPLE_RATE / freq
    for n in range(num_samples):
        # basic square wave
        if (n % period) < (period / 2):
            val = amplitude
        else:
            val = -amplitude
        # Apply slight envelope (attack/release)
        env = 1.0
        if n < 200:
            env = n / 200.0
        elif n > num_samples - 200:
            env = (num_samples - n) / 200.0
        samples.append(int(val * env))
    return samples

def write_wav(filename, samples):
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1) # mono
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        for s in samples:
            wav_file.writeframes(struct.pack('<h', s))

def generate_upbeat_menu_bgm():
    # Fun, bouncy major arpeggios (C major to F major)
    notes = [523.25, 659.25, 783.99, 659.25,  698.46, 880.00, 1046.50, 880.00] # C5, E5, G5, E5, F5, A5, C6, A5
    note_dur = 0.15
    
    track = []
    for _ in range(8):
        for freq in notes:
            track.extend(build_square_wave(freq, note_dur, AMPLITUDE * 0.5))
            
    bass_notes = [261.63, 0, 261.63, 0,  349.23, 0, 349.23, 0] # bouncy C3 / F3
    bass_track = []
    for _ in range(8):
        for freq in bass_notes:
            bass_track.extend(build_square_wave(freq, note_dur, AMPLITUDE * 0.6))
            
    # Mix
    mixed = []
    for t1, t2 in zip(track, bass_track):
        mixed.append(max(-32768, min(32767, t1 + t2)))
    write_wav("assets/menu_bgm.wav", mixed)

def generate_fun_game_bgm(fast=False):
    # D minor pentatonic fun, high energy progression
    # D5, F5, G5, A5, C6
    notes = [587.33, 698.46, 783.99, 880.00,  880.00, 783.99, 698.46, 587.33]
    
    # Bassline driving 1/8th notes
    bass_notes = [146.83, 146.83, 146.83, 146.83, 174.61, 196.00, 174.61, 146.83]
    
    note_dur = 0.11 if not fast else 0.075  # Huge tempo increase for fast mode
    
    track = []
    bass_track = []
    
    for _ in range(16):
        for freq in notes:
            track.extend(build_square_wave(freq, note_dur, AMPLITUDE * 0.4))
        for freq in bass_notes:
            bass_track.extend(build_square_wave(freq, note_dur, AMPLITUDE * 0.7))
            
    mixed = []
    for t1, t2 in zip(track, bass_track):
        mixed.append(max(-32768, min(32767, t1 + t2)))
            
    write_wav("assets/game_bgm_fast.wav" if fast else "assets/game_bgm.wav", mixed)

def main():
    os.makedirs("assets", exist_ok=True)
    print("Generating pure FUN assets/menu_bgm.wav...")
    generate_upbeat_menu_bgm()
    print("Generating pure FUN assets/game_bgm.wav...")
    generate_fun_game_bgm(fast=False)
    print("Generating pure FUN assets/game_bgm_fast.wav...")
    generate_fun_game_bgm(fast=True)
    print("Done!")

if __name__ == "__main__":
    main()
