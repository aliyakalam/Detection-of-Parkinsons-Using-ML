
import parselmouth

from parselmouth.praat import call

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    voicereport=parselmouth.praat.call([sound, pitch, pointProcess], "Voice report", 0, 10, 75, 600, 1.3, 1.6, 0.03, 0.45)
    dica = {}
    
    vr = voicereport.split("\n")
    vr = vr[1:]
    for line in vr:
      line = line.strip(" ")
      line = line.split(":")
      key = line[0]
      if len(line) > 1:
        temp = line[1]
        temp = temp.strip(" ")
        temp = temp.split(" ")[0]
        dica[key] = temp
    
    if(dica["Mean period"] == '--undefined--'):
        mean_period = 0
    else:
        mean_period = float(dica["Mean period"])
    standard_deviation_period = float(dica["Standard deviation of period"])
    ac = float(dica["Mean autocorrelation"])
    nth = float(dica["Mean noise-to-harmonics ratio"])
    htn = float(dica["Mean harmonics-to-noise ratio"])


    if dica["Fraction of locally unvoiced frames"][-1] == "%":
      fraction_unvoiced_frames = float(dica["Fraction of locally unvoiced frames"][:-1])*0.01
    else:
      fraction_unvoiced_frames = float(dica["Fraction of locally unvoiced frames"])
    num_voice_breaks = float(dica["Number of voice breaks"])

    if dica["Degree of voice breaks"][-1] == "%":  
      degree_voice_breaks = float(dica["Degree of voice breaks"][:-1])*0.01
    else:
      degree_voice_breaks = float(dica["Degree of voice breaks"])
    median_pitch = float(dica["Median pitch"])
    min_pitch = float(dica["Minimum pitch"])
    max_pitch = float(dica["Maximum pitch"])
    
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    n_pulses = parselmouth.praat.call(pulses, "Get number of points")
    n_periods = parselmouth.praat.call(pulses, "Get number of periods", 0.0, 0.0, 0.0001, 0.02, 1.3)
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, ac, nth, htn,median_pitch, meanF0, stdevF0, min_pitch, max_pitch, n_pulses, n_periods, mean_period, standard_deviation_period, fraction_unvoiced_frames, num_voice_breaks, degree_voice_breaks
 